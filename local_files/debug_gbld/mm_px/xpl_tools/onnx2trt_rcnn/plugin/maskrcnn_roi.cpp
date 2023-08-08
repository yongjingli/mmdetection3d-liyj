/*
 * Copyright (c) 2018, Xiaopeng. All rights reserved.
 * Create by caizw @ 2018.9.12 maskrcnn roi_net
 *
 *
 * Input: rpn_cls_probs_fpnX,rpn_bbox_pred_fpnX,im_info
 * -> GenerateProposalsOp : rpn_rois_fpnX, rpn_roi_probs_fpnX
 * -> CollectAndDistributeFpnRpnProposalsOp : rois, rois_fpnX
 * -> RoIAlign : roi_feat_fpnX
 * -> Concat : roi_feat_shuffled
 * -> BatchPermutation : roi_feat
 * -> FC -> Relu : fc6
 * -> FC -> Relu : fc7
 * fc7 -> FC : cls_score
 * -> SoftMax: cls_prob
 * fc7 -> FC : bbox_pred
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "../plugin/topKnms.h"
#include "MaskrcnnPlugin.hpp"

#define TEST_GPU_NMS 1

extern int prob_num;

typedef struct DATA {
  float *f;
  int shape[4];
} DATA;

typedef struct RECT {
  float left;
  float top;
  float right;
  float bottom;
} RECT;

typedef struct Tensor {
  const char *name;
  DATA *data;
} Tensor;

static int CollectAndDisOp(Tensor *input, Tensor *output);
static int GenerateProposalsOpGPU(Tensor *inputs, Tensor *outputs, int PM,
                                  int _feat_stride, int *nms_out, void *buffer,
                                  int blocksize, cudaStream_t cuda_stream);
static void GenerateAnchors(std::vector<int> an_sizes,
                            std::vector<int> an_stride,
                            std::vector<float> an_ratios, float *anchors);

static int CollectAndRoiAlign(Tensor *inputs, Tensor *outputsT,
                              const void *const *inputsGPU, void **outputsGPU,
                              void *workspace, cudaStream_t stream);
int RoiAlignWithFeatID(const void *const *featPtr, void **outputs,
                       int input_dims[], int roi_cols, int output_dims[],
                       void *workspace, cudaStream_t stream);

// 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR,-1:skip Plugin enqueue(for DLA)
int TRT_DEBUGLEVEL = 1;
// Support to change at converting and adjust at inference.
int pre_nms_topN = PRE_NMS_topN;
int post_nms_topN = POST_NMS_topN;
int CollectAndDisOpPlugin::initialize() {
  if (_initialized) {
    return 0;
  }

  {
    char *val = getenv("TRT_PRE_NMS");
    if (NULL != val) {
      pre_nms_topN = atoi(val);
      DPRINTF(1, "getenv TRT_PRE_NMS=%d\n", pre_nms_topN);
    }
  }

  auto const &input0_dims = this->getInputDims(0);
  auto const &input1_dims = this->getInputDims(1);

  pCPUInput0 = new float[input0_dims.d[0]];
  pCPUInput1 = new float[input1_dims.d[0]];
  pCPUOutput0 = new float[post_nms_topN * 5 * 5];  // post_nms_topN = 300
  pCPUOutput1 = new float[post_nms_topN];

#if TEST_GPU_NMS
  cudaMalloc(&pGPUBuffer0, PRE_NMS_topN * 5 * 5 * sizeof(float));
  cudaMalloc(&pGPUBuffer1, input0_dims.d[0] * sizeof(float));
  // Generate Anchors  and copy to Symbol memory on GPU
  ACNUM = anchor_ratios.size();
  SZNUM = anchor_sizes.size();
  GenerateAnchors(anchor_sizes, anchor_stride, anchor_ratios, pCPUOutput0);
  SetCudaSymbol(ACNUM, pCPUOutput0, 4 * SZNUM * ACNUM);
  // Generate Indices and copy to GPU, for sortTopK
  cudaMalloc(&pGPUIndices, 5 * input0_dims.d[0] * sizeof(int));
  for (int i = 0; i < input0_dims.d[0]; i++) ((int *)pCPUInput0)[i] = i;
  cudaMemcpy(pGPUIndices, pCPUInput0, input0_dims.d[0] * sizeof(int),
             cudaMemcpyHostToDevice);
#endif

  pCPUBuffer0 = new float[PRE_NMS_topN * 5 * 5];  // pre_nms_topN = 1000
  pCPUBuffer1 = new float[PRE_NMS_topN * 5];

  _initialized = true;

  return 0;
}

void CollectAndDisOpPlugin::terminate() {
  if (!_initialized) {
    return;
  }

  delete[] pCPUInput0;
  delete[] pCPUInput1;
  delete[] pCPUOutput0;
  delete[] pCPUOutput1;

#if TEST_GPU_NMS
  cudaFree(pGPUBuffer0);
  cudaFree(pGPUBuffer1);
  cudaFree(pGPUIndices);
#endif
  delete[] pCPUBuffer0;
  delete[] pCPUBuffer1;

  _initialized = false;
}

static float g_im_info[3] = {256, 640, 1.0};
int roialignsize[4 * 16] = {
    0,
};  // MaxBatchsize = 16;

void SetImInfoForCollect(float height64, float width64, float scale) {
  g_im_info[0] = height64;
  g_im_info[1] = width64;
  g_im_info[2] = scale;
}

inline int64_t volumeShape(const int d[], int nbDims = 4) {
  int64_t tot = 1;
  for (int i = 0; i < nbDims; i++)
    if (d[i] >= 0) tot *= d[i];
  return tot;
}

int CollectAndDisOpPlugin::enqueue(int batchSize, const void *const *inputs,
                                   void **outputs, void *workspace,
                                   cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip CollectAndDisOpPlugin::enqueue!!\n");
    return 0;
  }

  auto const &input0_dims = this->getInputDims(0);
  auto const &input1_dims = this->getInputDims(1);
  const int featUsed = anchor_stride.size();  // 5 or 4

  if (batchSize > 16) {
    batchSize = 16;
    DPRINTF(2, "Warning: CollectAndDisOp batchSize=%d BUT only process 16 \n",
            batchSize);
  }

  int input_batch_offset = 0;   // in floats
  int output_batch_offset = 0;  // in floats
  for (int nb = 0; nb < batchSize; nb++) {
    // prcoess batch size <=16. Cost at TX2: GPUsyc 20ms , CPU 15ms per batch
    memset(pCPUOutput0, 0, sizeof(float) * post_nms_topN * 5 * 5);
    int type = 0;
    if (getDataType() == nvinfer1::DataType::kFLOAT) {
      DPRINTF(2, "CollectAndDisOp fp32 input shape = %dx1, %dx1 pre_nms=%d\n",
              input0_dims.d[0], input1_dims.d[0], pre_nms_topN);
      type = 0;
    } else {
      DPRINTF(2, "CollectAndDisOp fp16 input shape = %dx1, %dx1 \n",
              input0_dims.d[0], input1_dims.d[0]);
      type = 1;
    }

    int roiNum = 0;
    int input_offset = 0;   // in floats
    int output_offset = 0;  // in floats
    {
      float *pGPUInput0 = (float *)inputs[0] + input_batch_offset;
      float *pGPUInput1 = (float *)inputs[1] + input_batch_offset * 4;
      // float im_info[3] = {height,1344.0,0.6942708492279053};
      DATA input2 = {g_im_info, {3, -1, -1, -1}};
      int num_out[5] = {
          0,
      };
      int nms_keep[5][PRE_NMS_topN];
      for (int i = 0; i < featUsed; i++) {
        int feat_stride = anchor_stride[i];
        int y = (int)g_im_info[0] / feat_stride;
        int x = (int)g_im_info[1] / feat_stride;

        DATA input0 = {pGPUInput0 + input_offset, {ACNUM, y, x, -1}};
        DATA input1 = {pGPUInput1 + input_offset * 4, {ACNUM * 4, y, x, -1}};
        input_offset += ACNUM * x * y;

        DATA output0 = {pGPUBuffer0 + output_offset * 5, {0, 0, 0, 0}};
        DATA output1 = {pGPUBuffer1 + output_offset, {0, 0, 0, 0}};

        Tensor inputsT[3] = {{"im_info", &input2},
                             {"rpn_cls_probs_fpn", &input0},
                             {"rpn_bbox_pred_fpn", &input1}};
        Tensor outputsT[2] = {{"rpn_rois_fpn", &output0},        // bbox
                              {"rpn_roi_probs_fpn", &output1}};  // score

        DPRINTF(2, "CollectAndDisOp ACNUM=%d, stride =%d input0={%d,%d,%d}\n",
                ACNUM, feat_stride, input0.shape[0], input0.shape[1],
                input0.shape[2]);
        num_out[i] = GenerateProposalsOpGPU(inputsT, outputsT, i, feat_stride,
                                            nms_keep[i], pGPUIndices,
                                            input0_dims.d[0], stream);
        DPRINTF(2, "CollectAndDisOp shape= %d(%d), %d \n",
                outputsT[0].data->shape[0], num_out[i],
                outputsT[0].data->shape[1]);

        if (post_nms_topN < num_out[i])
          num_out[i] = post_nms_topN;  // Only need post_nms_topN maximum
        for (int j = 0; j < num_out[i]; j++) {
          nms_keep[i][j] += output_offset;
        }
        output_offset += outputsT[0].data->shape[0];  //
      }
      convertCPU_GPU(pCPUBuffer0, pGPUBuffer0, output_offset * 5, type, 1,
                     stream, workspace);
      convertCPU_GPU(pCPUBuffer1, pGPUBuffer1, output_offset, type, 1, stream,
                     workspace);
      // for( int i = 0; i< output_offset; i+=16)printf("score[%d]=%f\n",i,
      // pCPUBuffer1[i] );
      {  // Keep NMS's result
        for (int i = 0; i < featUsed; i++) {
          for (int j = 0; j < num_out[i]; j++) {
            if (roiNum != nms_keep[i][j]) {
              memcpy(pCPUBuffer0 + roiNum * 5, pCPUBuffer0 + nms_keep[i][j] * 5,
                     sizeof(float) * 5);
              pCPUBuffer1[roiNum] = pCPUBuffer1[nms_keep[i][j]];
            }

            if (j < 10) {
              float *det_f = pCPUBuffer0 + roiNum * 5;
              DPRINTF(3, "GPU[%d]-> det[%d]:( %f, %f, %f, %f) score: %f\n",
                      nms_keep[i][j], roiNum, det_f[1], det_f[2], det_f[3],
                      det_f[4], pCPUBuffer1[roiNum]);
            }
            roiNum++;
          }
          DPRINTF(2, "GPU Keep NMS num_out[%d]=%d roiNum=%d\n", i, num_out[i],
                  roiNum);
        }
      }
    }

    if (hasROIAlign) {  // Combened with ROIAlign
      auto feat2_dims = this->getInputDims(2);
#if NV_TENSORRT_MAJOR == 6
      // fsd, add by dyg
      if (prob_num == 4) {
        feat2_dims.d[0] = 64;
      } else {
        feat2_dims.d[0] = 256;  // dims of plugin has error in TensorRT6.0.0
      }
      feat2_dims.d[1] = (int)g_im_info[1] / 4;
      feat2_dims.d[2] = (int)g_im_info[0] / 4;
#endif
      int feat_ch = feat2_dims.d[0];  // MODEL.RESNETS.RES2_OUT_CHANNELS
      DPRINTF(2, "Input[2] shape=%d(%d,%d,%d) nb=%d\n", feat2_dims.nbDims,
              feat2_dims.d[0], feat2_dims.d[1], feat2_dims.d[2], nb);

      DATA input0 = {pCPUBuffer0, {roiNum, 5, -1, -1}};  //"rpn_rois_fpn"
      DATA input1 = {pCPUBuffer1, {roiNum, 1, -1, -1}};  //"rpn_roi_probs_fpn"
      DATA input2 = {pCPUInput0,
                     {feat_ch, (int)g_im_info[0], (int)g_im_info[1], -1}};
      Tensor inputsT[3] = {{"rpn_rois_fpn", &input0},
                           {"rpn_roi_probs_fpn", &input1},
                           {"im_in", &input2}};  // feat info as im_in
      DATA output0 = {pCPUOutput0, {post_nms_topN, 5, -1, -1}};  //"rois" 100x5
      DATA output1 = {pCPUOutput1,
                      {post_nms_topN, feat_ch, _poolsize[0],
                       _poolsize[1]}};  // "feat_out" 100x256x7x7
      Tensor outputsT[2] = {{"rois", &output0}, {"feat_out", &output1}};

      const void *inGPU[10] = {inputs[0], inputs[1]};
      void *outGPU[2] = {outputs[0], outputs[1]};
      for (int i = 0; i < featUsed; i++) {
        int y = feat2_dims.d[1] / (1 << i);
        int x = feat2_dims.d[2] / (1 << i);
        inGPU[i + 2] = inputs[i + 2] + nb * feat_ch * x * y * sizeof(float);
        DPRINTF(2, "inGPU[%d] shape=%dx(%d,%d,%d)\n", i + 2, nb, feat_ch, y, x);
      }
      outGPU[0] += nb * volumeShape(output0.shape) * sizeof(float);
      outGPU[1] += nb * volumeShape(output1.shape) * sizeof(float);
      CollectAndRoiAlign(inputsT, outputsT, inGPU, outGPU, workspace, stream);
    } else {
      DATA input0 = {pCPUBuffer0, {roiNum, 5, -1, -1}};
      DATA input1 = {pCPUBuffer1, {roiNum, 1, -1, -1}};

      DATA output0 = {pCPUOutput0, {0, 0, 0, 0}};
      DATA output1 = {pCPUOutput0 + post_nms_topN * 5, {0, 0, 0, 0}};
      DATA output2 = {pCPUOutput0 + post_nms_topN * 2 * 5, {0, 0, 0, 0}};
      DATA output3 = {pCPUOutput0 + post_nms_topN * 3 * 5, {0, 0, 0, 0}};
      DATA output4 = {pCPUOutput0 + post_nms_topN * 4 * 5, {0, 0, 0, 0}};
      DATA output5 = {pCPUOutput1, {0, 1, 0, 0}};

      Tensor inputsT[2] = {{"rpn_rois_fpn", &input0},
                           {"rpn_roi_probs_fpn", &input1}};
      Tensor outputsT[6] = {
          {"rois", &output0},      {"rois_fpn2", &output1},
          {"rois_fpn3", &output2}, {"rois_fpn4", &output3},
          {"rois_fpn5", &output4}, {"rpn_roi_probs_fpn", &output5}};

      DPRINTF(2, "CollectAndDisOp input shape= %d, %d \n",
              inputsT[0].data->shape[0], inputsT[0].data->shape[1]);

      CollectAndDisOp(inputsT, outputsT);

      for (int i = 0; i < 6; i++) {
        DPRINTF(2, "CollectAndDisOp[%d] shape= %d, %d workspace=%p\n", i,
                outputsT[i].data->shape[0], outputsT[i].data->shape[1],
                workspace);
        convertCPU_GPU(outputsT[i].data->f,
                       (float *)outputs[i] +
                           output_batch_offset * outputsT[i].data->shape[1],
                       (post_nms_topN * outputsT[i].data->shape[1]), type, 0,
                       stream, workspace);
      }
      for (int i = 0; i < 4; i++) {
        roialignsize[i + 4 * nb] = outputsT[i + 1].data->shape[0];
      }
    }

    input_batch_offset += input_offset;
    output_batch_offset += post_nms_topN;
  }

  return 0;
}

std::vector<int> sort_indexes(const float v[], int vsize, int partsort = 0) {
  // initialize original index locations
  std::vector<int> idx(vsize);
  for (int i = 0; i < vsize; ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  if (partsort > 0 && partsort < vsize) {
    // std::partial_sort(idx.begin(), idx.begin() + partsort, idx.end(),
    //     [&v](int i1, int i2) {return v[i1] >  v[i2];});   // python version
    //     is not stable (quicksort)
    std::stable_sort(idx.begin(), idx.end(), [&v](int i1, int i2) {
      return v[i1] > v[i2];
    });  // and GPU version is stable
    idx.resize(partsort);
  } else {
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](int i1, int i2) { return v[i1] > v[i2]; });
  }

  return idx;
}

int mybbox_transform(std::vector<RECT> &boxes, DATA deltas, DATA pred_boxes,
                     std::vector<int> &order) {
  /*Forward transform that maps proposal boxes to predicted ground-truth
  boxes using bounding-box regression deltas. See bbox_transform_inv for a
  description of the weights argument.
  */
  if (order.size() == 0) {
    pred_boxes.shape[0] = 0;
    return 0;
  }

  // pred_boxe = np.zeros(4, dtype=deltas.dtype)
  // weights=(1.0, 1.0, 1.0, 1.0)
  float wx = 1.0f;
  float wy = 1.0f;
  float ww = 1.0f;
  float wh = 1.0f;

  int imgsize = deltas.shape[1] * deltas.shape[2];
  for (unsigned int oi = 0; oi < order.size(); oi++) {
    int i = order[oi];
    float width = boxes[i].right - boxes[i].left + 1.0f;
    float height = boxes[i].bottom - boxes[i].top + 1.0f;
    float ctr_x = boxes[i].left + 0.5f * width;
    float ctr_y = boxes[i].top + 0.5f * height;
    int id3 = i % imgsize, im3 = i / imgsize;
    float dx = deltas.f[im3 * 4 * imgsize + id3] / wx;
    float dy = deltas.f[(im3 * 4 + 1) * imgsize + id3] / wy;
    float dw = deltas.f[(im3 * 4 + 2) * imgsize + id3] / ww;
    float dh = deltas.f[(im3 * 4 + 3) * imgsize + id3] / wh;
    dw = std::min(dw, 4.135166556742356f);
    dh = std::min(dh, 4.135166556742356f);
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = expf(dw) * width;
    float pred_h = expf(dh) * height;

    pred_boxes.f[oi * 4 + 0] = pred_ctr_x - 0.5f * pred_w;
    pred_boxes.f[oi * 4 + 1] = pred_ctr_y - 0.5f * pred_h;
    pred_boxes.f[oi * 4 + 2] = pred_ctr_x + 0.5f * pred_w - 1;
    pred_boxes.f[oi * 4 + 3] = pred_ctr_y + 0.5f * pred_h - 1;

    if (oi < 10) {
      DPRINTF(3, "%d boxes[%d] = {%f,%f,%f,%f}\n", id3 * 3 + im3, i,
              boxes[i].left, boxes[i].top, boxes[i].right, boxes[i].bottom);
      DPRINTF(3, "ddeltas[%d] = {%f,%f,%f,%f}\n", i, dx, dy, dw, dh);
      DPRINTF(3, "pred_boxes[%u] = {%f,%f,%f,%f}\n", oi,
              pred_boxes.f[oi * 4 + 0], pred_boxes.f[oi * 4 + 1],
              pred_boxes.f[oi * 4 + 2], pred_boxes.f[oi * 4 + 3]);
    }
  }

  return 0;
}

int mynms(DATA &dets, DATA &scores, float thresh, int nms_topN) {
  float *x1 = &dets.f[0];
  float *y1 = &dets.f[1];
  float *x2 = &dets.f[2];
  float *y2 = &dets.f[3];
  int suppressed[PRE_NMS_topN];
  memset(suppressed, 0, sizeof(suppressed));
  float areas[PRE_NMS_topN];
  for (int i = 0; i < dets.shape[0]; i++) {
    float ix1 = x1[i * 4];
    float iy1 = y1[i * 4];
    float ix2 = x2[i * 4];
    float iy2 = y2[i * 4];
    if (ix1 < ix2 && iy1 < iy2)
      areas[i] = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);
    else {
      areas[i] = 0;
      suppressed[i] = 1;
    }
  }

  int roiIdx = 0;  // roi index post nms
  for (int i = 0; i < dets.shape[0]; i++) {
    if (suppressed[i] == 1) continue;
    float ix1 = x1[i * 4];
    float iy1 = y1[i * 4];
    float ix2 = x2[i * 4];
    float iy2 = y2[i * 4];
    // float iarea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);
    for (int tj = 0; tj < i; tj++) {
      if (suppressed[tj] == 1) continue;
      float jx1 = x1[tj * 4];
      float jy1 = y1[tj * 4];
      float jx2 = x2[tj * 4];
      float jy2 = y2[tj * 4];
      float xx1 = std::max(jx1, ix1);
      float yy1 = std::max(jy1, iy1);
      float xx2 = std::min(jx2, ix2);
      float yy2 = std::min(jy2, iy2);
      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      // float jarea = (jx2 - jx1 + 1) * (jy2 - jy1 + 1);
      // float ovr = inter / (areas[i] + areas[tj] - inter);
      if (inter > thresh * (areas[tj] + areas[i] - inter)) {
        suppressed[i] = 1;
        break;
      }
    }

    if (suppressed[i] == 0) {
      roiIdx++;
      if (roiIdx >= nms_topN) break;  // scores.f[i] < SCORE_THRESH/2 ||
    }
  }

  int j = 0;
  for (int i = 0; i < dets.shape[0]; i++) {
    if (0 == suppressed[i]) {
      if (i != j) {
        memcpy(&dets.f[j * 4], &dets.f[i * 4], sizeof(float) * 4);
        // memcpy(&scores.f[j], &scores.f[i], sizeof(float));
        memcpy(&scores.f[j * scores.shape[1]], &scores.f[i * scores.shape[1]],
               sizeof(float) * scores.shape[1]);  // add by dyg
      }
      DPRINTF(4, "det[%d]:( %f, %f, %f, %f) score: %f\n", j, dets.f[j * 4],
              dets.f[j * 4 + 1], dets.f[j * 4 + 2], dets.f[j * 4 + 3],
              scores.f[j * scores.shape[1]]);
      j++;
      if (j >= nms_topN) break;
    }
  }

  dets.shape[0] = j;  // j

  return 0;
}

//_generate_anchors( {4,8,16,32,64}, 5,  {0.1,0.2,0.3,0.4,0.5,0.6,1.0,2.0}, 8 );
static void GenerateAnchors(std::vector<int> an_sizes,
                            std::vector<int> an_stride,
                            std::vector<float> aspect_ratios, float *anchors) {
  const int sizeNum = an_sizes.size();
  const int aspectNum = aspect_ratios.size();

  /*Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
  */
  float *out = anchors;
  for (int j = 0; j < sizeNum; j++) {
    const int scales = an_sizes[j] / an_stride[j];
    int anchor[4] = {0, 0, an_stride[j] - 1, an_stride[j] - 1};
    for (int i = 0; i < aspectNum; i++) {
      float ratios = aspect_ratios[i];
      // Enumerate a set of anchors for each aspect ratio wrt an anchor.
      float w = anchor[2] - anchor[0] + 1;
      float h = anchor[3] - anchor[1] + 1;
      float x_ctr = anchor[0] + 0.5f * (w - 1);
      float y_ctr = anchor[1] + 0.5f * (h - 1);

      float size = w * h;
      float size_ratios = size / ratios;
      // round half to even, add by dyg
      float ws = rint(sqrt(size_ratios));
      float hs = rint(ws * ratios);

      ws *= scales;
      hs *= scales;

      out[0] = int(x_ctr - 0.5f * (ws - 1));
      out[1] = int(y_ctr - 0.5f * (hs - 1));
      out[2] = int(x_ctr + 0.5f * (ws - 1));
      out[3] = int(y_ctr + 0.5f * (hs - 1));
      out += 4;
    }
  }
  if (TRT_DEBUGLEVEL >= 4) {  // Check the anchor
    float *anchor = anchors;
    for (int j = 0; j < sizeNum; j++) {
      for (int i = 0; i < aspectNum; i++) {
        printf("[%d,%d] anchor(%f,%f,%f,%f)\n", j, i, anchor[0], anchor[1],
               anchor[2], anchor[3]);
        anchor += 4;
      }
    }
    // fflush(stdout);
    // exit(0);
  }
}

#if !TEST_GPU_NMS
#error "CPU version didn't support to generate anchors dynamically"
// Removed legacy code for CPU version by caizw @20191108
// static int GenerateProposalsOp(Tensor *inputs, Tensor *outputs,
//                               int _feat_stride)
#else
// GPU version for GenerateProposalsOp
static int GenerateProposalsOpGPU(Tensor *inputs, Tensor *outputs, int PM,
                                  int _feat_stride, int *nms_out, void *buffer,
                                  int blocksize, cudaStream_t cuda_stream) {
  // input image (height, width, scale), in which scale is the scale factor
  DATA im_info = *inputs[0].data;
  int im_height = (int)im_info.f[0];
  int im_width = (int)im_info.f[1];

  // predicted probability of fg object for each RPN anchor
  DATA scores = *inputs[1].data;
  int sc_height = scores.shape[1];
  int sc_width = scores.shape[2];
  // predicted achors transformations
  DATA bbox_deltas = *inputs[2].data;

  DATA *im_i_boxes = outputs[0].data;
  DATA *im_i_probs = outputs[1].data;
  im_i_probs->shape[1] = 1;

  // 4. sort all (proposal, score) pairs by score from highest to lowest
  // 5. take top pre_nms_topN (e.g. 6000)   pre_nms_topN= 1000
  int datasize = scores.shape[0] * scores.shape[1] * scores.shape[2];
  im_i_probs->shape[0] = datasize;
  if (im_i_probs->shape[0] > pre_nms_topN) {
    im_i_probs->shape[0] = pre_nms_topN;
  }
  int *indics = (int *)buffer;
  int *order = indics + blocksize;
  void *tempbuf = order + blocksize;
  size_t tempsize = blocksize * 3 * 4;
  // sortTopK(scores.f, datasize, im_i_probs->shape[0], pre_nms_topN,
  //         im_i_probs->f, order, cuda_stream);
  sortByKey(tempbuf, tempsize, scores.f, im_i_probs->f, indics, order, datasize,
            cuda_stream);
  DPRINTF(3, "Af sortTopK datasize=%d tmpsize=%lu\n", datasize,
          (sortByKey(nullptr, tempsize, scores.f, im_i_probs->f, indics, order,
                     datasize, cuda_stream),
           tempsize));

  // 1. Generate proposals from bbox deltas and shifted anchors
  // 2. clip proposals to image (may result in proposals with zero area
  // 3. remove predicted boxes with either height or width < min_size
  bool clip = true;
  if (prob_num > 1) clip = false;  // fsd not clip
  int boxnum = GetShiftedAnchors(
      im_i_probs->shape[0], im_height, im_width, sc_height, sc_width, PM,
      _feat_stride, order, bbox_deltas.f, im_i_boxes->f, cuda_stream, clip);
  DPRINTF(3, "GetShiftedAnchors boxnum=%d\n", boxnum);
  im_i_boxes->shape[0] = boxnum;
  im_i_boxes->shape[1] = 5;

  // 6. apply loose nms (e.g. threshold = 0.7)     nms_thresh = 0.7
  // 7. take after_nms_topN (e.g. 300)   post_nms_topN =   1000
  // 8. return the top proposals (-> RoIs top)
  int num_out = 0;
  if (nms_thresh > 0) {
    // cudaMemcpy( g_workspace, proposals.f, (j*4)*sizeof(float),
    // cudaMemcpyHostToDevice);

    nms(nms_out, &num_out, im_i_boxes->f, boxnum, 5, nms_thresh, order,
        cuda_stream);
    DPRINTF(3, "GPU nms num_out=%d\n", num_out);
  }
  return num_out;
}
#endif

static int CollectAndDisOp(Tensor *inputs, Tensor *outputs) {
  // rois = collect(inputs, self._train)
  // int post_nms_topN=1000;
  int k_max = 6;
  int k_min = 2;
  int lvl_min = 2;
  int lvl_max = 5;
  int s0 = 224;
  int lvl0 = 4;

  int num_klvls = k_max - k_min + 1;
  Tensor roi_inputs = inputs[0];  // rpn_rois_fpn2-6, already concatenated
  Tensor score_inputs =
      inputs[1];  // rpn_roi_probs_fpn2-6, already concatenated

  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  DATA rois = *roi_inputs.data;
  DATA scores = *score_inputs.data;
  std::vector<int> inds =
      sort_indexes(scores.f, scores.shape[0], post_nms_topN);
  // rois = rois[inds, :]
  // outputs[0].reshape(rois.shape)
  DATA &dst = *outputs[0].data;
  dst.shape[0] = (int)inds.size();
  dst.shape[1] = 5;
  for (int i = 0; i < dst.shape[0]; i++) {
    memcpy(&dst.f[i * 5], &rois.f[inds[i] * 5], 5 * sizeof(float));
  }
  rois = dst;

  // lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  for (int lvl = lvl_min; lvl <= lvl_max; lvl++) {
    outputs[lvl - 1].data->shape[0] = 0;
    outputs[lvl - 1].data->shape[1] = 5;
    // Fill the output buffers of fpn to avoid "uniformly zero", for int8
    // network calibration
    memmove(outputs[lvl - 1].data->f, rois.f,
            rois.shape[0] * 5 * sizeof(float));
  }

  for (int i = 0; i < rois.shape[0]; i++) {
    DPRINTF(3, "rois[%d]:( %.3f, %.3f, %.3f, %.3f)\n", i, rois.f[i * 5 + 1],
            rois.f[i * 5 + 2], rois.f[i * 5 + 3], rois.f[i * 5 + 4]);
    float w = (rois.f[i * 5 + 3] - rois.f[i * 5 + 1] + 1);
    float h = (rois.f[i * 5 + 4] - rois.f[i * 5 + 2] + 1);
    float areas = w * h;
    float s = sqrtf(areas);
    // Eqn.(1) in FPN paper
    int target_lvls = floor(lvl0 + log2(s / s0 + 1e-6));
    int lvl = (target_lvls < lvl_min)
                  ? lvl_min
                  : ((target_lvls > lvl_max) ? lvl_max : target_lvls);
    // int output_idx = lvl - 2;
    //# Create new roi blobs for each FPN level
    DATA *outspdata = outputs[lvl - 1].data;  // rois_fpn2-5
    int idx = outspdata->shape[0];
    memmove(&outspdata->f[idx * 5], &rois.f[i * 5], 5 * sizeof(float));
    outspdata->f[idx * 5] = i;
    outspdata->shape[0]++;
  }

  DATA *outidx = outputs[num_klvls].data;
  outidx->shape[0] = post_nms_topN;
  for (int lvl = lvl_min; lvl <= lvl_max; lvl++) {
    DATA *outspdata = outputs[lvl - 1].data;  // rois_fpn2-5
    int idxoffset = (lvl - lvl_min) *
                    post_nms_topN;  // different with python version. Fixed
    // offset of fpn layer to post_nms_topN.
    for (int j = 0; j < outspdata->shape[0]; j++) {
      outidx->f[(int)outspdata->f[j * 5]] = idxoffset++;
      outspdata->f[j * 5] += 1;
    }
  }

  for (int i = 0; i < outidx->shape[0]; i++) {
    DPRINTF(3, "outidx[%d] = %.1f\n", i, outidx->f[i]);
  }
  return 0;
}

static int CollectAndRoiAlign(Tensor *inputs, Tensor *outputsT,
                              const void *const *inputsGPU, void **outputsGPU,
                              void *workspace, cudaStream_t stream) {
  // rois = collect(inputs, self._train)
  int lvl_min = 2;
  int lvl_max = 5;
  int s0 = 224;
  int lvl0 = 4;

  Tensor roi_inputs = inputs[0];    // rpn_rois_fpn2-6, already concatenated
  Tensor score_inputs = inputs[1];  // rpn_roi_probs_fpn2-6, concatenated

  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  DATA rois = *roi_inputs.data;
  DATA scores = *score_inputs.data;
  std::vector<int> inds =
      sort_indexes(scores.f, scores.shape[0], post_nms_topN);
  // rois = rois[inds, :]
  // outputs[0].reshape(rois.shape)
  DATA &dst = *outputsT[0].data;
  dst.shape[0] = (int)inds.size();  // shape[1] = 5;
  for (int i = 0; i < dst.shape[0]; i++) {
    memcpy(&dst.f[i * 5], &rois.f[inds[i] * 5], 5 * sizeof(float));
  }
  rois = dst;

  // lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  for (int i = 0; i < rois.shape[0]; i++) {
    float w = (rois.f[i * 5 + 3] - rois.f[i * 5 + 1] + 1);
    float h = (rois.f[i * 5 + 4] - rois.f[i * 5 + 2] + 1);
    float areas = w * h;
    float s = sqrtf(areas);
    // Eqn.(1) in FPN paper
    int target_lvls = floor(lvl0 + log2(s / s0 + 1e-6));
    int lvl = (target_lvls < lvl_min)
                  ? lvl_min
                  : ((target_lvls > lvl_max) ? lvl_max : target_lvls);
    rois.f[i * 5] = lvl;  // [2,5]
    DPRINTF(3, "rois[%d]:(%.3f,%.3f,%.3f,%.3f) lv:%d\n", i, rois.f[i * 5 + 1],
            rois.f[i * 5 + 2], rois.f[i * 5 + 3], rois.f[i * 5 + 4], lvl);
  }

  // copy rois to GPU
  convertCPU_GPU(rois.f, outputsGPU[0], (post_nms_topN * rois.shape[1]), 0, 0,
                 stream, workspace);

  // RoiAlign using fpn_levels information in rois[:, 0]
  {
    DATA *imIn = inputs[2].data;       // 256,256,640
    DATA *featOut = outputsT[1].data;  // 100,256,7,7
    RoiAlignWithFeatID(inputsGPU, outputsGPU, imIn->shape, rois.shape[1],
                       featOut->shape, workspace, stream);
  }

  return 0;
}

static int onebbox_transform(float *boxes, float *deltas, float *result,
                             float *pWeight, float *im_info) {
  if (boxes == nullptr || deltas == nullptr || result == nullptr ||
      pWeight == nullptr)
    return -1;

  // weights=(1.0, 1.0, 1.0, 1.0)
  float wx = pWeight[0];
  float wy = pWeight[1];
  float ww = pWeight[2];
  float wh = pWeight[3];

  float width = boxes[2] - boxes[0] + 1.0f;
  float height = boxes[3] - boxes[1] + 1.0f;
  float ctr_x = boxes[0] + 0.5f * width;
  float ctr_y = boxes[1] + 0.5f * height;
  {
    float dx = deltas[0] / wx;
    float dy = deltas[1] / wy;
    float dw = deltas[2] / ww;
    float dh = deltas[3] / wh;
    dw = std::min(dw, 4.135166556742356f);
    dh = std::min(dh, 4.135166556742356f);
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = expf(dw) * width;
    float pred_h = expf(dh) * height;

    float left = pred_ctr_x - 0.5f * pred_w;
    float top = pred_ctr_y - 0.5f * pred_h;
    float right = pred_ctr_x + 0.5f * pred_w - 1;
    float bottom = pred_ctr_y + 0.5f * pred_h - 1;
#if 0  // clip_image
    // float im_scale = im_info[2];
    float im_height = im_info[0];
    float im_width = im_info[1];

    if (left < 0) left = 0;
    if (left > im_width - 1) left = im_width - 1;

    if (right < 0) right = 0;
    if (right > im_width - 1) right = im_width - 1;

    if (top < 0) top = 0;
    if (top > im_height - 1) top = im_height - 1;
    if (bottom < 0) bottom = 0;
    if (bottom > im_height - 1) bottom = im_height - 1;
#endif
    result[0] = left;
    result[1] = top;
    result[2] = right;
    result[3] = bottom;
    DPRINTF(3, "deltas = {%f,%f,%f,%f}\n", deltas[0], deltas[1], deltas[2],
            deltas[3]);
    // printf( "pred_boxes[%d] = {%f,%f,%f,%f}\n", oi,
    // pred_boxes.f[oi*4+0],pred_boxes.f[oi*4+1],pred_boxes.f[oi*4+2],pred_boxes.f[oi*4+3]
    // );
  }

  return 0;
}

// function 'box_results_with_nms_and_limit'
// "rois" -> bbox_transform -> nms -> FPN level -> "mask_rois_fpnX"
static int pre_masknet(Tensor *inputs, Tensor *outputs, float *im_info) {
  // box_results_with_nms_and_limit
  DATA *boxes = inputs[0].data;   // "gpu_0/rois",{post_nms_topN, 5}
  DATA *deltas = inputs[1].data;  // "gpu_0/bbox_pred",{post_nms_topN,24}
  DATA *scores = inputs[2].data;  // "gpu_0/cls_prob",{post_nms_topN,6}
  DATA *occupy = inputs[3].data;  // "gpu_0/occupy_prob",{post_nms_topN,2}
  DATA *lever = inputs[4].data;   // "gpu_0/lever_prob",{post_nms_topN,2}
  DATA *lock = inputs[5].data;    // "gpu_0/lock_prob",{post_nms_topN,2}

  for (int inds = 0; inds < boxes->shape[0] * boxes->shape[1]; inds++) {
    boxes->f[inds] /= im_info[2];
  }

  // add by dyg
  int box_prob_num = 4 + prob_num;  // [score, x, y, w, h, occupy, lever, lock]
  bool is_fsd = prob_num > 1 ? true : false;

  int num_classes = scores->shape[1];  // 6
  std::vector<DATA> cls_boxes(num_classes);
  std::vector<DATA> cls_scores(num_classes);
  int classIndex[DETECTIONS_PER_IM];
  float weight[] = {10.0f, 10.0f, 5.0f, 5.0f};
  // Apply threshold on detection probabilities and apply NMS
  // Skip j = 0, because it's the background class
  for (int j = 1; j < num_classes; j++) {
    float *dets = new float[4 * post_nms_topN];
    float tmpdets[4 * post_nms_topN];
    float tmpscore1[post_nms_topN];
    // add by dyg
    float tmpoccupy1[post_nms_topN];
    float tmplever1[post_nms_topN];
    float tmplock1[post_nms_topN];
    float *detscores = new float[DETECTIONS_PER_IM * prob_num];
    DATA scoreclass = {detscores, {DETECTIONS_PER_IM, prob_num, 0, 0}};
    int dets_inds = 0;
    for (int inds = 0; inds < scores->shape[0]; inds++) {
      float score = scores->f[inds * num_classes + j];
      if (score > SCORE_THRESH) {
        tmpscore1[dets_inds] = score;
        // add by dyg
        if (is_fsd) {
          // if shape[1] is 2, use last value
          int last_idx = occupy->shape[1] - 1;
          int len = occupy->shape[1];
          tmpoccupy1[dets_inds] = occupy->f[inds * len + last_idx];
          tmplever1[dets_inds] = lever->f[inds * len + last_idx];
          tmplock1[dets_inds] = lock->f[inds * len + last_idx];
        }
        DPRINTF(3, "Befor bbox_transform %d boxes[%d] = {%f,%f,%f,%f}\n", j,
                inds, boxes->f[inds * 5 + 1], boxes->f[inds * 5 + 2],
                boxes->f[inds * 5 + 3], boxes->f[inds * 5 + 4]);
        onebbox_transform(&boxes->f[inds * 5 + 1],
                          &deltas->f[(inds * num_classes + j) * 4],
                          &tmpdets[dets_inds * 4], weight, im_info);
        DPRINTF(
            3, "After onebbox_transform %d dets[%d] = {%f,%f,%f,%f} score=%f\n",
            j, dets_inds, tmpdets[dets_inds * 4], tmpdets[dets_inds * 4 + 1],
            tmpdets[dets_inds * 4 + 2], tmpdets[dets_inds * 4 + 3], score);
        dets_inds++;
      }
    }
    DATA dets_j = {dets, {dets_inds, 4, -1, -1}};
    if (dets_inds > 0) {
      std::vector<int> order =
          sort_indexes(tmpscore1, dets_inds, DETECTIONS_PER_IM);
      scoreclass.shape[0] = order.size();
      dets_j.shape[0] = order.size();
      for (int i = 0; i < scoreclass.shape[0]; i++) {
        int oi = order[i];
        scoreclass.f[i * prob_num] = tmpscore1[oi];
        // add by dyg
        if (is_fsd) {
          scoreclass.f[i * prob_num + 1] = tmpoccupy1[oi];
          scoreclass.f[i * prob_num + 2] = tmplever1[oi];
          scoreclass.f[i * prob_num + 3] = tmplock1[oi];
        }
        memcpy(&dets_j.f[i * 4], &tmpdets[oi * 4], sizeof(float) * 4);
      }
      DPRINTF(3, "Befor mynms %d dets_j.shape[0] = %d \n", j, dets_j.shape[0]);
      mynms(dets_j, scoreclass, NMS, DETECTIONS_PER_IM);
      DPRINTF(3, "After mynms %d dets_j.shape[0] = %d \n", j, dets_j.shape[0]);
    }

    cls_boxes[j] = dets_j;
    cls_scores[j] = scoreclass;
    // need to delete dets_j at the end;
  }
  // Limit to max_per_image detections **over all classes**
  /*
  if cfg.TEST.DETECTIONS_PER_IM > 0:    #100
   */

  DATA &rois = *outputs[0].data;
  int rois_inds = 0;
  for (int j = 1; j < num_classes; j++) {
    for (int i = 0; i < cls_boxes[j].shape[0]; i++) {
      if (rois_inds >= DETECTIONS_PER_IM)
        break;  // TODO: caizhiwen sort by score

      rois.f[rois_inds * box_prob_num] =
          cls_scores[j].f[i * prob_num] * 0.1f;  // must < 1.0f (now 0.1 max )
      for (int c = 0; c < 4; c++) {
        rois.f[rois_inds * box_prob_num + 1 + c] =
            cls_boxes[j].f[i * 4 + c] * im_info[2];  //
      }
      // occupy/lever/lock, add by dyg
      for (int c = 0; c < prob_num - 1; c++) {
        rois.f[rois_inds * box_prob_num + 5 + c] =
            cls_scores[j].f[i * prob_num + 1 + c];
      }
      classIndex[rois_inds] = j;
      rois_inds++;
    }
    delete[] cls_boxes[j].f;   // delete  dets
    delete[] cls_scores[j].f;  // delete detscores
  }
  rois.shape[0] = rois_inds;

  int lvl_min = 2;
  int lvl_max = 5;
  int s0 = 224;
  int lvl0 = 4;
  int num_lvls = 5;

  for (int lvl = lvl_min; lvl <= lvl_max; lvl++) {
    outputs[lvl - 1].data->shape[0] = 0;
    outputs[lvl - 1].data->shape[1] = 5;
  }

  DATA *outidx = outputs[num_lvls].data;
  for (int i = 0; i < rois.shape[0]; i++) {
    float w = (rois.f[i * box_prob_num + 3] - rois.f[i * box_prob_num + 1] + 1);
    float h = (rois.f[i * box_prob_num + 4] - rois.f[i * box_prob_num + 2] + 1);
    float areas = w * h;
    float s = sqrt(areas);
    // Eqn.(1) in FPN paper
    int target_lvls = floor(lvl0 + log2(s / s0 + 1e-6));
    int lvl = (target_lvls < lvl_min)
                  ? lvl_min
                  : ((target_lvls > lvl_max) ? lvl_max : target_lvls);
    int output_idx = lvl - 2;
    //# Create new roi blobs for each FPN level
    DATA *outspdata = outputs[output_idx + 1].data;  // rois_fpn2-5
    int idx = outspdata->shape[0];
    memcpy(&outspdata->f[idx * 5], &rois.f[i * box_prob_num],
           5 * sizeof(float));
    outspdata->f[idx * 5] = i + 0.001 * classIndex[i];
    outspdata->shape[0]++;
    outspdata->shape[1] = 5;

    rois.f[i * box_prob_num] +=
        classIndex[i];  // mark the class tpye in rois data, easy for output.
    DPRINTF(2, "No.%d class:%d rois:(%.3f, %.3f, %.3f, %.3f, %.3f) \n", i,
            classIndex[i], rois.f[i * box_prob_num],
            rois.f[i * box_prob_num + 1], rois.f[i * box_prob_num + 2],
            rois.f[i * box_prob_num + 3], rois.f[i * box_prob_num + 4]);
    if (is_fsd) {
      DPRINTF(2, "occupy:%.3f lever:%.3f lock:%.3f\n",
              rois.f[i * box_prob_num + 5], rois.f[i * box_prob_num + 6],
              rois.f[i * box_prob_num + 7]);
    }
  }
  if (rois.shape[0] < DETECTIONS_PER_IM)
    rois.f[rois.shape[0] * box_prob_num] =
        0;  // mark the end with default type 0 :background;

  outidx->shape[0] = 0;
  for (int lvl = lvl_min; lvl <= lvl_max; lvl++) {
    int output_idx = lvl - 2;
    DATA *outspdata = outputs[output_idx + 1].data;  // rois_fpn2-5
    for (int j = 0; j < outspdata->shape[0]; j++) {
      outidx->f[(int)outspdata->f[j * 5]] = outidx->shape[0]++;
    }
  }

  for (int i = 0; i < outidx->shape[0]; i++) {
    DPRINTF(3, "outidx[%d] = %.1f\n", i, outidx->f[i]);
  }
  return 0;
}

/*
"rois"/"bbox_pred"/"cls_prob" ->	bbox_transform -> nms -> FPN level ->
"mask_rois_fpnX" "mask_rois_fpnX" ->	RoiAlign -> Concat -> BatchPermutation
-> "_[mask]_rois_feat"
*/
int RunPreMasknet(float **inputDatas, float **outDatas, int *roiNums,
                  int classNum, int width, int height, float im_scale) {
  if (nullptr == inputDatas || nullptr == outDatas || nullptr == roiNums)
    return -1;

  float im_info[] = {(float)height, (float)width, im_scale};
  DATA input0 = {inputDatas[0], {post_nms_topN, 5, -1, -1}};  //"rois"
  DATA input1 = {inputDatas[1],
                 {post_nms_topN, classNum * 4, -1, -1}};  //"bbox_pred"
  DATA input2 = {inputDatas[2],
                 {post_nms_topN, classNum, -1, -1}};  //"cls_prob"
  // add by dyg
  int prob_class_num = 2;  // TODO: should be 1
  DATA input3 = {inputDatas[3],
                 {post_nms_topN, prob_class_num, -1, -1}};  //"occupy_prob"
  DATA input4 = {inputDatas[4],
                 {post_nms_topN, prob_class_num, -1, -1}};  //"lever_prob"
  DATA input5 = {inputDatas[5],
                 {post_nms_topN, prob_class_num, -1, -1}};  //"lock_prob"
  // inputDatas[3] ~ inputDatas[6]
  // fpn_res2_2_sum\fpn_res3_3_sum\fpn_res4_5_sum\fpn_res5_2_sum

  DATA output0 = {outDatas[0], {0, 0, 0, 0}};
  DATA output1 = {outDatas[1], {0, 0, 0, 0}};
  DATA output2 = {outDatas[2], {0, 0, 0, 0}};
  DATA output3 = {outDatas[3], {0, 0, 0, 0}};
  DATA output4 = {outDatas[4], {0, 0, 0, 0}};
  DATA output5 = {outDatas[5], {0, 0, 0, 0}};

  Tensor inputs[6] = {{"rois", &input0},       {"bbox_pred", &input1},
                      {"cls_prob", &input2},   {"occupy_prob", &input3},
                      {"lever_prob", &input4}, {"lock_prob", &input5}};
  Tensor outputs[6] = {
      {"mask_rois", &output0},      {"mask_rois_fpn2", &output1},
      {"mask_rois_fpn3", &output2}, {"mask_rois_fpn4", &output3},
      {"mask_rois_fpn5", &output4}, {"mask_rois_idx_restore_int32", &output5}};

  int ret = pre_masknet(inputs, outputs, im_info);

  for (int i = 0; i < 6; i++) {
    DATA *outspdata = outputs[i].data;
    roiNums[i] = outspdata->shape[0];
    DPRINTF(3, "output[%d] shape={%d,%d}\n", i, outspdata->shape[0],
            outspdata->shape[1]);
  }

  return ret;
}
