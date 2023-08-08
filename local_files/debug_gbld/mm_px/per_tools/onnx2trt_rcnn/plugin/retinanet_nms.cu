/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12 for retinaNet
 *
 * Input: class(5),box(5)
 * input -> Decode -> Concat -> NMS -> output
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "RetinaNetPlugin.hpp"

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

namespace retinacuda {
// refer from nvidia retinanet-example
#include <cstdint>
#include <stdexcept>

#define CUDA_ALIGN 256

__constant__ float4 g_WT;
__constant__ float4 g_ANCHORS[5 * 9];

void SetCudaSymbol(float *weight, float *anchorBuf, int anNum) {
  cudaMemcpyToSymbol(g_WT, weight, sizeof(float) * 4, 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g_ANCHORS, anchorBuf, sizeof(float) * anNum, 0,
                     cudaMemcpyHostToDevice);
}

//_generate_anchors( {4,8,16,32,64}, {32,64,128,356,512}, {0.5,1.0,2.0},
//{1.0,1.414,2} );
void GenerateAnchors(std::vector<int> an_sizes, std::vector<int> an_stride,
                     std::vector<float> aspect_ratios,
                     std::vector<float> scale_octave, float *anchors) {
  const int sizeNum = an_sizes.size();
  const int aspectNum = aspect_ratios.size();
  const int octiveNum = scale_octave.size();

  /*Generate anchor (reference) windows by enumerating aspect ratios X
  scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
  */
  float *out = anchors;
  for (int j = 0; j < sizeNum; j++) {
    for (int i = 0; i < aspectNum; i++) {
      for (int k = 0; k < octiveNum; k++) {
        float an_size = an_sizes[j] * scale_octave[k];
        const float scales = an_size / an_stride[j];
        int anchor[4] = {0, 0, an_stride[j] - 1, an_stride[j] - 1};
        float ratios = aspect_ratios[i];
        // Enumerate a set of anchors for each aspect ratio wrt an anchor.
        float w = anchor[2] - anchor[0] + 1;
        float h = anchor[3] - anchor[1] + 1;
        float x_ctr = anchor[0] + 0.5f * (w - 1);
        float y_ctr = anchor[1] + 0.5f * (h - 1);

        float size = w * h;
        float size_ratios = size / ratios;
        float ws = round(sqrtf(size_ratios) + 1e-5f);
        float hs = round(ws * ratios + 1e-5);

        ws *= scales;
        hs *= scales;

        out[0] = (x_ctr - 0.5f * (ws - 1));
        out[1] = (y_ctr - 0.5f * (hs - 1));
        out[2] = (x_ctr + 0.5f * (ws - 1));
        out[3] = (y_ctr + 0.5f * (hs - 1));

        out += 4;
      }
    }
  }
  if (TRT_DEBUGLEVEL >= 4) {  // Check the anchor
    float *anchor = anchors;
    for (int j = 0; j < sizeNum; j++) {
      for (int i = 0; i < aspectNum * octiveNum; i++) {
        DPRINTF(3, "[%d,%d] anchor(%f,%f,%f,%f)\n", j, i, anchor[0], anchor[1],
                anchor[2], anchor[3]);
        anchor += 4;
      }
    }
  }
}

template <typename T>
inline size_t get_size_aligned(size_t num_elem) {
  size_t size = num_elem * sizeof(T);
  size_t extra_align = 0;
  if (size % CUDA_ALIGN != 0) {
    extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
  }
  return size + extra_align;
}

template <typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace,
                       size_t &workspace_size) {
  size_t size = get_size_aligned<T>(num_elem);
  if (size > workspace_size) {
    throw std::runtime_error("Workspace is too small!");
  }
  workspace_size -= size;
  T *ptr = reinterpret_cast<T *>(workspace);
  workspace =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
  return ptr;
}

int decode(int batch, const void *const *inputs, void **outputs, size_t height,
           size_t width, size_t scale, size_t num_anchors, size_t num_classes,
           int anchors_offset, float score_thresh, int top_n, void *workspace,
           size_t workspace_size, cudaStream_t stream) {
  int scores_size = num_anchors * num_classes * height * width;

  auto anchors_d =
      get_next_ptr<float>(num_anchors * 4, workspace, workspace_size);

  auto on_stream = thrust::cuda::par.on(stream);

  auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
  auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto indices_sorted =
      get_next_ptr<int>(scores_size, workspace, workspace_size);
  auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
  auto scores_sorted =
      get_next_ptr<float>(scores_size, workspace, workspace_size);

  auto in_scores = static_cast<const float *>(inputs[0]) + batch * scores_size;
  auto in_boxes = static_cast<const float *>(inputs[1]) +
                  batch * (scores_size / num_classes) * 4;

  auto out_scores = static_cast<float *>(outputs[0]);
  auto out_boxes = static_cast<float4 *>(outputs[1]);
  auto out_classes = static_cast<float *>(outputs[2]);

  // Discard scores below threshold
  thrust::transform(on_stream, in_scores, in_scores + scores_size, flags,
                    thrust::placeholders::_1 > score_thresh);

  int *num_selected = reinterpret_cast<int *>(indices_sorted);
  thrust::cuda_cub::cub::DeviceSelect::Flagged(
      workspace, workspace_size,
      thrust::cuda_cub::cub::CountingInputIterator<int>(0), flags, indices,
      num_selected, scores_size, stream);
  cudaStreamSynchronize(stream);
  int num_detections = *thrust::device_pointer_cast(num_selected);
  // printf("scale=%d th=%f num_detections=%d\n", scale, score_thresh,
  // num_detections );
  // Only keep top n scores
  auto indices_filtered = indices;
  if (num_detections > top_n) {
    thrust::gather(on_stream, indices, indices + num_detections, in_scores,
                   scores);
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
        workspace, workspace_size, scores, scores_sorted, indices,
        indices_sorted, num_detections, 0, sizeof(*scores) * 8, stream);
    indices_filtered = indices_sorted;
    num_detections = top_n;
  }
  const float max_w = width * scale - 1.0f;
  const float max_h = height * scale - 1.0f;
  // Gather boxes
  thrust::transform(
      on_stream, indices_filtered, indices_filtered + num_detections,
      thrust::make_zip_iterator(
          thrust::make_tuple(out_scores, out_boxes, out_classes)),
      [=] __device__(int i) {
        int ix = i % width;
        int iy = (i / width) % height;
        int a = (i / num_classes / height / width) % num_anchors;
        int cls = (i / height / width) % num_classes;
        float dx = in_boxes[((a * 4 + 0) * height + iy) * width + ix] / g_WT.x;
        float dy = in_boxes[((a * 4 + 1) * height + iy) * width + ix] / g_WT.y;
        float dw = in_boxes[((a * 4 + 2) * height + iy) * width + ix] / g_WT.z;
        float dh = in_boxes[((a * 4 + 3) * height + iy) * width + ix] / g_WT.w;

        // Add anchors offsets to deltas
        float x = ix * scale;
        float y = iy * scale;
        const float4 &d = g_ANCHORS[anchors_offset + a];

        float x1 = x + d.x;
        float y1 = y + d.y;
        float x2 = x + d.z;
        float y2 = y + d.w;
        float w = x2 - x1 + 1.0f;
        float h = y2 - y1 + 1.0f;
        float pred_ctr_x = dx * w + x1 + 0.5f * w;
        float pred_ctr_y = dy * h + y1 + 0.5f * h;
        float hpred_w = expf(fminf(dw, 4.135166556742356f)) * w * 0.5f;
        float hpred_h = expf(fminf(dh, 4.135166556742356f)) * h * 0.5f;
        // printf("%d %d(%f,%f,%f,%f)]\n", __LINE__, i, x1,y1,x2,y2);
        // printf("%d %d(%f,%f,%f,%f)]\n",__LINE__, i, dx,dy,dw,dh);
        float4 box = float4{fmaxf(0.0f, pred_ctr_x - hpred_w),
                            fmaxf(0.0f, pred_ctr_y - hpred_h),
                            fminf(pred_ctr_x + hpred_w - 1.0f, max_w),
                            fminf(pred_ctr_y + hpred_h - 1.0f, max_h)};
        // printf("%d %d(%f,%f,%f,%f)]\n", __LINE__, i,
        // box.x,box,y,box.z,box.w);

        return thrust::make_tuple(in_scores[i], box, cls);
      });

  return num_detections;
}

__global__ void nms_kernel(const int num_per_thread, const float threshold,
                           const int num_detections, const int *indices,
                           float *scores, const float *classes,
                           const float4 *boxes) {
  // Go through detections by descending score
  for (int m = 0; m < num_detections; m++) {
    int max_idx = indices[m];
    int mcls = classes[max_idx];
    float4 mbox = boxes[max_idx];
    // printf("%d(%f,%f,%f,%f)]\n", max_idx, mbox.x,mbox.y,mbox.z,mbox.w);
    for (int n = 0; n < num_per_thread; n++) {
      int i = threadIdx.x * num_per_thread + n;
      if (i < num_detections && m < i && scores[m] > 0.0f) {
        int idx = indices[i];
        int icls = classes[idx];
        if (mcls == icls) {
          float4 ibox = boxes[idx];
          float x1 = max(ibox.x, mbox.x);
          float y1 = max(ibox.y, mbox.y);
          float x2 = min(ibox.z, mbox.z);
          float y2 = min(ibox.w, mbox.w);
          float w = max(0.0f, x2 - x1 + 1);
          float h = max(0.0f, y2 - y1 + 1);
          float iarea = (ibox.z - ibox.x + 1) * (ibox.w - ibox.y + 1);
          float marea = (mbox.z - mbox.x + 1) * (mbox.w - mbox.y + 1);
          float inter = w * h;
          float overlap = inter / (iarea + marea - inter);
          if (overlap > threshold) {
            scores[i] = 0.0f;
          }
        }
      }
    }

    // Sync discarded detections
    __syncthreads();
  }
}

int nms(int batch, const float *const *inputs, void **outputs, size_t count,
        int detections_per_im, float nms_thresh, void *workspace,
        size_t workspace_size, cudaStream_t stream) {
  auto on_stream = thrust::cuda::par.on(stream);

  auto flags = get_next_ptr<bool>(count, workspace, workspace_size);
  auto indices = get_next_ptr<int>(count, workspace, workspace_size);
  auto indices_sorted = get_next_ptr<int>(count, workspace, workspace_size);
  auto scores = get_next_ptr<float>(count, workspace, workspace_size);
  auto scores_sorted = get_next_ptr<float>(count, workspace, workspace_size);

  {
    auto in_scores = inputs[0];
    auto in_boxes = (const float4 *)(inputs[1]);
    auto in_classes = inputs[2];

    auto out_scores =
        static_cast<float *>(outputs[0]) + batch * detections_per_im;
    auto out_boxes =
        static_cast<float4 *>(outputs[1]) + batch * detections_per_im;
    auto out_classes =
        static_cast<float *>(outputs[2]) + batch * detections_per_im;

    int num_detections = count;
    // Gen indices [ 0...num_detections ] for SortPairsDescending
    thrust::sequence(on_stream, indices, indices + num_detections);

    // Sort scores and corresponding indices
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
        workspace, workspace_size, in_scores, scores_sorted, indices,
        indices_sorted, num_detections, 0, sizeof(*scores) * 8, stream);

    // Launch actual NMS kernel - 1 block with each thread handling n detections
    const int max_threads = 64;
    int num_per_thread = ceil((float)num_detections / max_threads);
    nms_kernel<<<1, max_threads, 0, stream>>>(
        num_per_thread, nms_thresh, num_detections, indices_sorted,
        scores_sorted, in_classes, in_boxes);

    // Re-sort with updated scores
    thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
        workspace, workspace_size, scores_sorted, scores, indices_sorted,
        indices, num_detections, 0, sizeof(*scores) * 8, stream);

    // Gather filtered scores, boxes, classes
    num_detections = min(detections_per_im, num_detections);
    cudaMemcpyAsync(out_scores, scores, num_detections * sizeof *scores,
                    cudaMemcpyDeviceToDevice, stream);
    DPRINTF(2, "in:count=%lu out:num_detections=%d\n", count, num_detections);
    if (num_detections < detections_per_im) {
      thrust::fill_n(on_stream, out_scores + num_detections,
                     detections_per_im - num_detections, 0);
    }
    thrust::gather(on_stream, indices, indices + num_detections, in_boxes,
                   out_boxes);
    thrust::gather(on_stream, indices, indices + num_detections, in_classes,
                   out_classes);
  }

  return 0;
}

typedef struct boxdata34 {
  float data[34];
} BOX34;
typedef struct boxdata36 {
  float data[36];
} BOX36;

// optimized for MOD of xpmodel
template <typename T>
int filteByThresh(int batch, const void *inputbuf, void *outputbuf,
                  const nvinfer1::Dims dim, float thresh, void *workspace,
                  size_t ws_size, cudaStream_t stream) {
  const int max_num = 1000;
  const float log_conf_th = log(thresh / (1 - thresh)) * 0.999;
  auto on_stream = thrust::cuda::par.on(stream);
  size_t box_num = dim.d[0];
  int box_col = dim.d[1];
  DPRINTF(3, "%dx%d BOXSize T=%zu\n", box_num, box_col, sizeof(T));
  auto flags = get_next_ptr<bool>(box_num, workspace, ws_size);
  auto indices = get_next_ptr<int>(box_num, workspace, ws_size);
  int *num_selected = get_next_ptr<int>(4, workspace, ws_size);
  auto boxes_buf = get_next_ptr<T>(max_num, workspace, ws_size);
  int num_detections = 0;
  for (int nb = 0; nb < batch; nb++) {
    auto in_boxes = static_cast<const T *>(inputbuf) + nb * box_num;
    auto out_boxes = static_cast<T *>(outputbuf) + nb * box_num;
    int *out_idx = (int *)out_boxes;
    // Discard scores below threshold
    thrust::transform(on_stream, in_boxes, in_boxes + box_num, flags,
                      [=] __device__(T in) {
                        if (in.data[4] > log_conf_th)
                          return true;
                        else
                          return false;
                      });

    // int *num_selected = reinterpret_cast<int *>(flags);
    thrust::cuda_cub::cub::DeviceSelect::Flagged(
        workspace, ws_size,
        thrust::cuda_cub::cub::CountingInputIterator<int>(0), flags, indices,
        num_selected, box_num, stream);
    cudaMemcpyAsync(out_idx, indices, max_num * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    num_detections = *thrust::device_pointer_cast(num_selected);
    DPRINTF(2, "MOD col=%d th=%f num_detections=%d\n", box_col, thresh,
            num_detections);
    if (num_detections > max_num) num_detections = max_num;

    // Only keep top n scores
    thrust::gather(on_stream, indices, indices + num_detections, in_boxes,
                   boxes_buf);
    for (int i = 0; i < box_num; i++) {
      out_boxes[i].data[4] = -1.f;
    }

    for (int i = num_detections - 1; i >= 0; i--) {
      int idx = out_idx[i];
      DPRINTF(3, "out_idx[%d]=%d\n", i, idx);
      if (idx < 0 || idx >= box_num) continue;
      cudaMemcpyAsync(out_boxes + idx, boxes_buf + i, box_col * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
    }
  }

  return num_detections;
}

extern "C" int filteMOD(int batch, const void *inputbuf, void *outputbuf,
                        const nvinfer1::Dims dim, float thresh, void *workspace,
                        size_t ws_size, cudaStream_t stream) {
  if (34 == dim.d[1]) {
    return filteByThresh<BOX34>(batch, inputbuf, outputbuf, dim, thresh,
                                workspace, ws_size, stream);
  } else if (36 == dim.d[1]) {
    return filteByThresh<BOX36>(batch, inputbuf, outputbuf, dim, thresh,
                                workspace, ws_size, stream);
  } else
    return -2;
}

}  // namespace retinacuda

int DecodeAndNMSPlugin::initialize() {
  if (_initialized) {
    return 0;
  }

  auto const &input0_dims = this->getInputDims(0);
  auto const &input1_dims = this->getInputDims(1);

  _count = 10;
  DPRINTF(1, "NMSPlugin %d %ld %ld\n", _detections_per_im, _count,
          getWorkspaceSize(1));

  std::vector<int> anchor_sizes = {32, 64, 128, 256, 512};
  std::vector<int> anchor_stride = {8, 16, 32, 64, 128};
  std::vector<float> anchor_ratios = {0.5f, 1.0f, 2.0f};
  std::vector<float> scale_octave = {1.f, powf(2.f, 1.f / 3.f),
                                     powf(2.f, 2.f / 3.f)};
  int SZNUM = anchor_sizes.size();                         // 5;
  int ACNUM = anchor_ratios.size() * scale_octave.size();  // 3*3;
  float *pCPUOutput0 = new float[SZNUM * ACNUM * 4];
  // Generate Anchors  and copy to Symbol memory on GPU
  retinacuda::GenerateAnchors(anchor_sizes, anchor_stride, anchor_ratios,
                              scale_octave, pCPUOutput0);
  float weight[4] = {10.f, 10.f, 5.f, 5.f};
  retinacuda::SetCudaSymbol(weight, pCPUOutput0, SZNUM * ACNUM * 4);
  delete[] pCPUOutput0;

  _initialized = true;

  return 0;
}

void DecodeAndNMSPlugin::terminate() {
  if (!_initialized) {
    return;
  }

  _initialized = false;
}

int DecodeAndNMSPlugin::enqueue(int batchSize, const void *const *inputs,
                                void **outputs, void *workspace,
                                cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip CollectAndDisOpPlugin::enqueue!!\n");
    return 0;
  }

  auto const &input0_dims = this->getInputDims(0);
  auto const &input1_dims = this->getInputDims(1);

  if (batchSize > 16) {
    batchSize = 16;
    DPRINTF(2,
            "Warning: DecodeAndNMSPlugin batchSize=%d BUT only process 16 \n",
            batchSize);
  }

  for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    float *tmpbufer = (float *)((char *)workspace + decodeBufsize);
    float *inputs_nms[3] = {tmpbufer, tmpbufer + top_n * 5,
                            tmpbufer + top_n * 5 * 5};
    int num_decode = 0;
    int _count = _input_dims.size();
    for (int i = 0; i < _count / 2; i++) {
      auto const &scores_dims = _input_dims[i];
      auto const &boxes_dims = _input_dims[i + _count / 2];
      int height = scores_dims.d[1];
      int width = scores_dims.d[2];
      int num_anchors = boxes_dims.d[0] / 4;
      int num_classes = scores_dims.d[0] / num_anchors;
      float scale = _scales[i];
      const void *inputs_decode[2] = {inputs[i], inputs[i + _count / 2]};
      void *outputs_decode[3] = {inputs_nms[0] + num_decode,
                                 inputs_nms[1] + num_decode * 4,
                                 inputs_nms[2] + num_decode};
      DPRINTF(3, "DecodePlugin[%d]: %f %d %d %d %d\n", i, scale, height, width,
              num_anchors, num_classes);
      num_decode += retinacuda::decode(batchIdx, inputs_decode, outputs_decode,
                                       height, width, scale, num_anchors,
                                       num_classes, i * num_anchors, 0.3f,
                                       top_n, workspace, decodeBufsize, stream);
    }

    retinacuda::nms(batchIdx, inputs_nms, outputs, num_decode,
                    _detections_per_im, _nms_thresh, workspace, decodeBufsize,
                    stream);
  }
  return 0;
}
