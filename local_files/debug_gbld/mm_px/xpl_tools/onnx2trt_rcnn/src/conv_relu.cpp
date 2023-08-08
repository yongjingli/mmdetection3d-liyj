/* For mask-net of maskrcnn
 * Copyright (c) 2018, Xiaopeng. All rights reserved.
 * Create by caizw @ 2018.9.20
 * details of RunMaskNet:
 * "rois"/"bbox_pred"/"cls_prob" ->	bbox_transform -> nms -> FPN level ->
 * "mask_rois_fpnX" ->	RoiAlign -> Concat -> BatchPermutation
 * -> "_[mask]_rois_feat"
 * "_[mask]_rois_feat" -> Conv/Relu -> Conv/Relu -> Conv/Relu ->
 * ConvTranspose/Relu -> Conv/Sigmoid -> "mask_fcn_probs"
 */

#include <cudnn.h>
#include <npp.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>  // Chengzhang added
using namespace std::chrono;

#define CHECK(status)                           \
  {                                             \
    if (status != 0) {                          \
      DPRINTF(1, "Cuda failure: %d\n", status); \
      abort();                                  \
    }                                           \
  }

// Create MaskNet with TensorRT
#include "MaskrcnnPlugin.hpp"
#include "NvInfer.h"
#include "conv_relu.h"
using namespace nvinfer1;

// normal is 1: cls_prob. fsd is 4: cls_prob/occupy_prob/lever_prob/lock_prob.
// add by dyg
int prob_num = 1;

extern size_t ReadBinFile(std::string filename, char *&databuffer);
static void MaskPostProcessing(void *dMask, void *dOutput, void *contours,
                               int *roiOrder, nvinfer1::Dims &buffer_dim,
                               int detNum, float *detBox);
static void KeypointPostProcessing(void *dMask, void *dOutput, void *contours,
                                   int *roiOrder, int classNum, int detNum,
                                   float *detBox);

inline int64_t volume(const nvinfer1::Dims &d) {
  int64_t nSize = d.d[0];
  for (int i = 1; i < d.nbDims; i++) nSize *= d.d[i];
  return nSize;
}

// Logger for TensorRT info/warning/errors
class iLogger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING) DPRINTF(1, "[Mask]%s\n", msg);
  }
} gLoggerMask;

static int g_nOutLayer;
static void *g_dbuffer[3];  // device address
static void *g_hbuffer;     // host address
static void *g_dinput = nullptr;
static IRuntime *g_infer = nullptr;
// RunTime Structure for Mask & Keypoint branch
typedef struct {
  ICudaEngine *engine;
  IExecutionContext *context;
  int nbBindings;  // <=32
  int nMaxBatchsize;
  void *buffers[32];
  // cudaStream_t stream;
  int64_t bufferSizes[32];
  nvinfer1::Dims bufferDims[32];
  int nConvNum;  // number of conv layers, mask:4, keypoint: 8
 public:
  bool isInited() { return (nullptr != engine); }
  int initBranch(ICudaEngine *engine, int maxbatch);
  int doInferenceBranch(float *input, float *output, int batchSize,
                        cudaStream_t stream);
  void releaseBranch();
} MaskRunTime;
static MaskRunTime g_MaskRT[2];

int MaskRunTime::initBranch(ICudaEngine *inengine, int maxbatch) {
  if (nullptr == inengine) return -1;

  engine = inengine;
  // Run the engine to check.
  doInferenceBranch(nullptr, nullptr, maxbatch,
                    nullptr);  // also add g_nOutLayer

  return 0;
}

// ch: channle ID for engine, 0: mask-branch, 1: keypoint-branch
// inputType: 0:cpu float32, 1:gpu float32,
int MaskRunTime::doInferenceBranch(float *input, float *output, int batchSize,
                                   cudaStream_t stream) {
  if (nullptr == engine) return -1;

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  if (0 == nbBindings && batchSize > 0) {
    nMaxBatchsize = batchSize;
    nbBindings = engine->getNbBindings();
    if (nbBindings > 32) nbBindings = 32;  // <=32

    int kp_num = 0;

    // Create GPU buffers on device
    for (int i = 0; i < nbBindings; ++i) {
      bool isInput = engine->bindingIsInput(i);
      Dims dims = engine->getBindingDimensions(i);
      const char *name = engine->getBindingName(i);
      bufferSizes[i] = volume(dims) * sizeof(float);
      bufferDims[i] = dims;
      if (isInput) {  // shared g_dinput for two engiens
        if (nullptr == g_dinput) {
          CHECK(cudaMalloc(&g_dinput, bufferSizes[i] * batchSize));
        }
        buffers[i] = g_dinput;
      } else {
        CHECK(cudaMalloc(&buffers[i], bufferSizes[i] * batchSize));
        if (NULL != strstr(name, "keypoint") || NULL != strstr(name, "lever") ||
            NULL != strstr(name, "lock")) {
          nConvNum = 8;
          kp_num += dims.d[0];
        } else if (NULL != strstr(name, "mask")) {
          nConvNum = 4;
        } else if (1 == i) {
          nConvNum = 1;
        }
      }
      const char *pMark = (isInput ? "In" : "Out");
      DPRINTF(1, "MaskNet[%d] %s:%s [%d,%d,%d],%ld\n", i, pMark, name,
              dims.d[0], dims.d[1], dims.d[2], sizeof(float));
    }

    char databuf[33];
    snprintf(databuf, sizeof(databuf), "%d", kp_num);
    setenv("TRT_KPNUM", databuf, 0);

    return 0;
  }

  const int inputIndex = 0;
  // const int outputIndex = 1;
  if (batchSize > 0 && nullptr != stream) {
    if (nullptr == context) {
      context = engine->createExecutionContext();
    }

    // DMA input batch data to device, infer on the batch asynchronously, and
    // DMA output back to host
    if (nullptr != input)
      CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                            batchSize * bufferSizes[inputIndex],
                            cudaMemcpyHostToDevice, stream));
    // context->execute(batchSize, buffers);
    context->enqueue(batchSize, buffers, stream, nullptr);
    if (nullptr != output) {
      // add by dyg
      int output_offset = 0;
      for (int i = 0; i < nbBindings; i++) {
        if (!engine->bindingIsInput(i)) {
          int size = batchSize * bufferSizes[i];
          CHECK(cudaMemcpyAsync(output + output_offset, buffers[i], size,
                                cudaMemcpyDeviceToHost, stream));
          output_offset += size / sizeof(float);
        }
      }
    }
    cudaStreamSynchronize(stream);
  }

  if (batchSize <= 0 && context != nullptr && stream != nullptr) {
    context->destroy();
    context = nullptr;
    if (nullptr != g_dinput) {  // shared g_dinput for two engiens
      CHECK(cudaFree(g_dinput));
      g_dinput = nullptr;
    }
    for (int i = 0; i < nbBindings; i++) {
      if (!engine->bindingIsInput(i)) {
        CHECK(cudaFree(buffers[i]));
      }
    }
    nMaxBatchsize = 0;
    nbBindings = 0;
    nConvNum = 0;
  }

  return 0;
}

void MaskRunTime::releaseBranch() {
  if (nullptr != engine) {
    doInferenceBranch(nullptr, nullptr, 0, nullptr);
    engine->destroy();
    engine = nullptr;
  }
}

//"_[mask]_rois_feat"
// -> Conv/Relu -> Conv/Relu -> Conv/Relu -> ConvTranspose/Relu -> Conv/Sigmoid
// -> "mask_fcn_probs"
int InitMaskNetTRT(void *pWeight, int size, int InputSize, int maxbatch,
                   int classNum, std::string engine_file = "masknet.trt") {
  if (nullptr == pWeight || maxbatch > 100 || engine_file.empty()) return -1;

  int ch = 0;  // 0: mask-branch or single keypoint-branch, 1: keypoint-branch

  if (!g_MaskRT[ch].isInited()) {
    // If no engine then Try to load directly from serialized engine file(.trt)
    std::ifstream file(engine_file, std::ios::binary);
    if (file.good()) {
      std::vector<char> trtModelStream;
      size_t trtSize{0};

      file.seekg(0, file.end);
      trtSize = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream.resize(trtSize);
      file.read(trtModelStream.data(), trtSize);
      file.close();

      g_infer = createInferRuntime(gLoggerMask);
      char *pBuf = trtModelStream.data();
      for (int i = 0; i < 2 && trtSize > 0; i++) {
        int nEngineLen = *(int *)pBuf;
#if NV_TENSORRT_MAJOR == 6  // trt format changed @ TensorRT 6
        if (0 != memcmp("ptrt", pBuf, 4)) continue;
        nEngineLen = *(int *)(pBuf + 12) + 0x18;
#endif
        auto engine = g_infer->deserializeCudaEngine(pBuf, nEngineLen, nullptr);
        if (0 == g_MaskRT[i].initBranch(engine, maxbatch)) {
          DPRINTF(1, "CH%d deserialize %dB, tot %ldB\n", i, nEngineLen,
                  trtSize);
          g_nOutLayer += g_MaskRT[i].nConvNum;
        }

        pBuf += nEngineLen;
        trtSize -= nEngineLen;
      }
    }
  }

  // If no engine then Try to convert from weight
  if (!g_MaskRT[ch].isInited()) {
    // MaskNet changed, code removed @ 20191127
  }

  // Still no engien, return error
  if (!g_MaskRT[ch].isInited()) return -1;

  return 0;
}

//----------------------------------------------------------------------------
int InitMaskNet(void *pWeightfile, int InputSize, int maxbatch, int classNum) {
  std::string engine_file = (char *)pWeightfile;
  size_t size = 0;
  char *weightStream{nullptr};
  if (engine_file.length() < 255 && engine_file.length() > 4) {
    size = ReadBinFile(engine_file, weightStream);
    engine_file.replace(engine_file.end() - 4, engine_file.end(), ".trt");
    DPRINTF(1, "ReadBinFile for %s size=%ld Bytes\n", engine_file.c_str(),
            size);
  }

  if (nullptr == weightStream) return -1;

  char *val = getenv("TRT_CLASS");
  if (NULL != val) {
    classNum = atoi(val);
    printf("getenv TRT_CLASS=%d\n", classNum);
  }

  // InputSize for mask : 256*14*14*sizeof(float)=200704Byte, maxbatch<=100
  int ret = InitMaskNetTRT(weightStream, size, InputSize, maxbatch, classNum,
                           engine_file);

  delete[] weightStream;

  {  // alloc device buffer for Mask & Keypoint PostProcessing
    NppiSize oSizeROI = {640, 240};
    int nBufferSize = 0;
    nppiMaxIndxGetBufferHostSize_32f_C1R(oSizeROI, &nBufferSize);
    DPRINTF(1, "nppiMaxIndxGetBuffer = %d\n", nBufferSize);  // 2892 Byte

    const int bufsize[3] = {64 * 64 * 400, 640 * 240 * 40, 640 * 240 * 8};
    for (int i = 0; i < 3; i++) {
      CHECK(cudaMalloc(&g_dbuffer[i], bufsize[i]));  // float
      cudaMemset(g_dbuffer[i], 0, bufsize[i]);
    }
    // alloc host buffer for Keypoint PostProcessing
    g_hbuffer = malloc(64 * 64 * 100);  // uint8
    memset(g_hbuffer, 0, 64 * 64 * 100);
  }

  return ret;
}

int ForwardMaskNet(int ch, void *pInput, int InputSize, int nbatch,
                   void *pOutput, int OutputSize, cudaStream_t stream) {
  if (nbatch <= 0) return -1;

  if (-2 == TRT_DEBUGLEVEL) {  // save kp_feat for Int8Calibrator
    int insize = g_MaskRT[ch].bufferSizes[0];
    static int roiIdx = 0;
    char featfile[32];
    roiIdx++;
    for (int nb = 0; nb < nbatch; nb++) {
      CHECK(cudaMemcpyAsync(g_hbuffer, g_MaskRT[ch].buffers[1], insize,
                            cudaMemcpyDeviceToHost, stream));
      cudaStreamSynchronize(stream);
      {
        const char *ch_head[2] = {"mkfeat", "kpfeat"};
        snprintf(featfile, 31, "feat/%s_%d_%d.raw", ch_head[ch], roiIdx, nb);
        std::ofstream file(featfile, std::ios::out | std::ios::binary);
        file.write((char *)g_hbuffer, insize);
        file.close();
      }
    }
  }

  DPRINTF(2, "ForwardMaskNet%d nb=%d OutputSize=%d\n", ch, nbatch, OutputSize);
  g_MaskRT[ch].doInferenceBranch((float *)pInput, (float *)pOutput, nbatch,
                                 stream);
  return 0;
}

int DestroyMaskNet() {
  if (0 == g_nOutLayer) return 0;

  g_nOutLayer = 0;
  DPRINTF(2, "DestroyMaskNet\n");
  g_MaskRT[0].releaseBranch();
  g_MaskRT[1].releaseBranch();

  for (int i = 0; i < 3; i++)
    if (nullptr != g_dbuffer[i]) {
      CHECK(cudaFree(g_dbuffer[i]));
      g_dbuffer[i] = nullptr;
    }
  if (nullptr != g_hbuffer) {
    free(g_hbuffer);
    g_hbuffer = nullptr;
  }

  if (nullptr != g_infer) {
    g_infer->destroy();
    g_infer = nullptr;
  }

  return 0;
}

//------------------------------------------------------------------------------
static int g_height = 1208;
static int g_width = 1920;
static float g_im_scale = 0.69;
const int MaskSize = 28;
const int ID_FREESPACE = 5;

void SetImInfo(int imageHeight, int imageWidth, int height64, int width64,
               float im_scale) {
  g_height = imageHeight;
  g_width = imageWidth;
  g_im_scale = im_scale;
  // g_maxWidth = maxWidth;

  SetImInfoForCollect(height64, width64, im_scale);
  DPRINTF(1, "SetImInfoForCollect=( %d, %d, %f )\n", height64, width64,
          im_scale);
}

// inputs = model_net_result "rois"/"bbox_pred"/"cls_prob" as input for masknet
void RunMaskNet(std::vector<void *> &inputsGPU, nvinfer1::Dims *bufferDims,
                float *output, void *outputGPU, cudaStream_t stream) {
  int width = g_width;
  int height = g_height;
  float im_scale = g_im_scale;
  float *inputDatas[6];  // CPU
  float *outDatas[6];    // CPU, for "mask_rois" & "mask_rois_fpnX"
  void *outDataMask;     // CPU, for "mask_fcn_probs"
  int roiNums[6];        // inverse of BatchPermutation
  int classNum = bufferDims[2 + 2].d[1];  // "cls_prob" d[1] = 6
  int roisorder[DETECTIONS_PER_IM];

  time_point<std::chrono::high_resolution_clock> t_start;
  DPRINTF(2, "RunMaskNet Start %p\n",
          &(t_start = high_resolution_clock::now()));
  {  //"rois"/"bbox_pred"/"cls_prob"["occupy_prob_"/"lever_prob_"/"lock_prob_"]
    // -> bbox_transform -> nms -> FPN level -> "mask_rois_fpnX"
    // copy the outputs of faster-rcnn from CPU to GPU
    // {rois,bbox_pred,cls_prob[occpupy_prob_,lever_prob,lock_prob]}
    int offset = 0;
    int skip = 2;
    for (int i = 0; i < prob_num + 2; i++) {
      if (3 == i) skip = 6;
      int insizes =
          bufferDims[skip + i].d[0] *
          bufferDims[skip + i].d
              [1];  // sizeof
                    // "rois"/"bbox_pred"/"cls_prob"/"occupy_prob"/"lever_prob"/"lock_prob"
      inputDatas[i] = output + offset;
      CHECK(cudaMemcpyAsync(inputDatas[i], inputsGPU[i + skip],
                            insizes * sizeof(float), cudaMemcpyDeviceToHost,
                            stream));
      offset += insizes;

      const char *filename[6] = {"rois.log",       "bbox_pred.log",
                                 "cls_prob.log",   "occupy_prob.log",
                                 "lever_prob.log", "lock_prob.log"};
      // printf("save[%d] %s num=%d\n", i, filename[i], insizes);
      // saveFloatData(filename[i], (float*)inputDatas[i], insizes);
      DPRINTF(3, "%s={%f, %f, %f ...%f, %f, %f}\n", filename[i],
              inputDatas[i][0], inputDatas[i][1], inputDatas[i][2],
              inputDatas[i][insizes - 3], inputDatas[i][insizes - 2],
              inputDatas[i][insizes - 1]);
    }

    // int outsizes[5] = {DETECTIONS_PER_IM*5,DETECTIONS_PER_IM};
    for (int i = 0; i < 6; i++) {
      outDatas[i] = output + offset;
      if (0 == i) {
        offset += DETECTIONS_PER_IM * (prob_num + 4);
      } else if (i < 5) {
        offset += DETECTIONS_PER_IM * 5;
      } else {
        offset += DETECTIONS_PER_IM;
      }
    }
    outDataMask = output + offset;

    cudaStreamSynchronize(stream);

    //-> bbox_transform -> nms -> FPN rois
    RunPreMasknet((float **)inputDatas, outDatas, roiNums, classNum, width,
                  height, im_scale);
    DPRINTF(2, "Tot time of RunPreMasknet = %fms\n",
            duration<float, std::milli>(high_resolution_clock::now() - t_start)
                .count());
  }

  {
    float *pData = (float *)outDataMask;
    pData[0] = -10;                               // end for mask contour;
    pData[MaskSize * MaskSize * classNum] = -10;  // end for keypoint;
  }
  int roiAlignSize = 0;
  int detNum = roiNums[5];  // the finally number of detected objects.
  if (detNum <= 0 || 0 == g_nOutLayer) return;  // No MaskNet

  {  //"mask_rois_fpnX" ->	RoiAlign -> Concat -> BatchPermutation ->
    //"_[mask]_rois_feat"
    const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
    int sampling_ratio_ = 2;
    int pooled_height_ = 14;
    int pooled_width_ = 14;

    float spatial_scale[] = {1 / 4.0f, 1 / 8.0f, 1 / 16.0f, 1 / 32.0f};
    void *gpuBuffer1 =
        inputsGPU[5];  // TODO:not 2???  using buffer[5] gpu_0/rois_1[300, 5,
                       //  1] as workspace buffer (100*5)
    void *gpuBuffer2 = g_dinput;  // as workspace buffer (100*256*14*14) 20MB ?
    nvinfer1::Dims input_dims;
    nvinfer1::Dims roi_dims;
    nvinfer1::Dims output_dims;

    for (int i = 0; i < detNum; i++) {  // inverse of BatchPermutation
      roisorder[i] = (int)outDatas[5][i];
      DPRINTF(3, "mask fpn %d -> roisorder[%d] \n", (int)outDatas[5][i], i);
    }

    for (int fpn_idx = 2; fpn_idx <= 5; fpn_idx++) {
      if (roiNums[fpn_idx - 1] == 0) {
        continue;
      }

      void *mask_rois_fpn = outDatas[fpn_idx - 1];  // mask_rois_fpn2-5
      roi_dims.d[0] = roiNums[fpn_idx - 1];
      roi_dims.d[1] = 5;

      // copy every fpn layer of mask_rois data from CPU to GPU
      CHECK(cudaMemcpyAsync(gpuBuffer1, mask_rois_fpn,
                            roi_dims.d[0] * roi_dims.d[1] * sizeof(float),
                            cudaMemcpyHostToDevice, stream));

      void *inputsGPURoi[2] = {inputsGPU[fpn_idx + 3], gpuBuffer1};
      void *outputsGPURoi[1] = {gpuBuffer2};
      input_dims = bufferDims[fpn_idx + 3];
      output_dims = input_dims;
      output_dims.nbDims = 4;
      output_dims.type[3] = output_dims.type[2];
      output_dims.d[0] = roi_dims.d[0];
      output_dims.d[1] = input_dims.d[0];  // 256
      output_dims.d[2] = pooled_height_;
      output_dims.d[3] = pooled_width_;

      RoiAlignForward(dataType, inputsGPURoi, outputsGPURoi, input_dims,
                      roi_dims, output_dims, spatial_scale[fpn_idx - 2],
                      sampling_ratio_, gpuBuffer1, stream);

      DPRINTF(2, "MaskNet fpn[%d] RoiAlignForward finished ret = %d\n", fpn_idx,
              cudaGetLastError());

      roiAlignSize = output_dims.d[1] * output_dims.d[2] * output_dims.d[3] *
                     sizeof(float);
      gpuBuffer2 += roiAlignSize * roi_dims.d[0];
    }
  }

  {
    // "_[mask]_rois_feat" -> Conv/Relu -> Conv/Relu -> Conv/Relu ->
    // ConvTranspose/Relu -> Conv/Sigmoid -> "mask_fcn_probs"
    int insize = g_MaskRT[0].bufferSizes[0] * detNum;
    int outsize = g_MaskRT[0].bufferSizes[1] * detNum;
    float *pData = (float *)outDataMask;
    float *pDataEnd = pData + insize / sizeof(float);
    DPRINTF(2, "Mask Input[0..%d]={%f, %f, %f, ... %f, %f, %f }\n",
            roiNums[5] - 1, pData[0], pData[1], pData[2], pDataEnd[-3],
            pDataEnd[-2], pDataEnd[-1]);
    int kpNum = detNum;    // number of rois for keypoint processing
    int maskNum = detNum;  // number of rois for mask processing
    int maskIdx = detNum;  // start index of roi for mask processing
#if 1  // find the index of freesapce, splite rois to (keypoint + mask)
    int numFreeSpace = 0;
    if (8 + 4 <= g_nOutLayer) {                 // mask & keypoint
      for (int i = 0; i < detNum; i++) {        // inverse of BatchPermutation
        int classID = int(outDatas[0][i * 5]);  // classID + 0.1f*roiScore
        if (ID_FREESPACE == classID) {
          numFreeSpace++;
          if (1 == numFreeSpace) maskIdx = i;
        }
      }
      maskNum = numFreeSpace;
      kpNum = maskIdx;
      g_MaskRT[0].buffers[0] = g_dinput + roiAlignSize * roisorder[maskIdx];
      roisorder[maskIdx] = 0;
    }
#endif

    // device ptr g_MaskRT[ch].buffers[0](g_dinput) & g_MaskRT[ch].buffers[1]
    int ret =
        ForwardMaskNet(0, nullptr, insize, maskNum, pData, outsize, stream);

    // mask head is exported into main onnx model, add by dyg
    if (inputsGPU.size() == 13) {
      int noOrder[] = {0};
      float noBox[] = {0};
      MaskPostProcessing(inputsGPU[12], outputGPU, pData, noOrder,
                         bufferDims[12], 1, noBox);
    }

    if (0 == ret) {
      DPRINTF(2, "Mask Output[0..%d]={%f, %f, %f, ... %f, %f, %f }\n",
              maskNum - 1, pData[0], pData[1], pData[2], pDataEnd[-3],
              pDataEnd[-2], pDataEnd[-1]);
      nppSetStream(stream);
      if (8 == g_nOutLayer) {
        pData = (float *)outDataMask + MaskSize * MaskSize * classNum;
        int kp_offset = 0;
        // for park/lever/lock, add by dyg
        for (int i = 0; i < g_MaskRT[0].nbBindings; i++) {
          if (!g_MaskRT[0].engine->bindingIsInput(i)) {
            int kpClassNum = g_MaskRT[0].bufferDims[i].d[0];
            KeypointPostProcessing(g_MaskRT[0].buffers[i], outputGPU,
                                   pData + kp_offset, roisorder, kpClassNum,
                                   kpNum, outDatas[0]);
            kp_offset += kpNum * kpClassNum * 2;
          }
        }
      } else
        MaskPostProcessing(g_MaskRT[0].buffers[1], outputGPU, pData,
                           roisorder + maskIdx, g_MaskRT[0].bufferDims[1],
                           maskNum, outDatas[0] + maskIdx * 5);
    }

    if (8 + 4 <= g_nOutLayer) {  // mask & keypoint
      pData = (float *)outDataMask + MaskSize * MaskSize * classNum;
      int kpClassNum = g_MaskRT[1].bufferDims[1].d[0];
      int ret = ForwardMaskNet(1, nullptr, insize, kpNum, nullptr, outsize,
                               stream);  // device ptr g_dinput & g_doutput[ch]
      if (0 == ret) {
        KeypointPostProcessing(g_MaskRT[1].buffers[1], outputGPU, pData,
                               roisorder, kpClassNum, kpNum, outDatas[0]);
      }
    }
  }
}

static inline void interpolateCubic(float x, float *coeffs) {
  const float A = -0.75f;
  coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
  coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
  coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
  coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
  DPRINTF(4, "interpolateCubic x=%f c=%f,%f,%f,%f\n", x, coeffs[0], coeffs[1],
          coeffs[2], coeffs[3]);
}

void KeypointPostProcessing(void *dMask, void *dOutput, void *keypoints,
                            int *roiOrder, int classNum, int kpMaskNum,
                            float *detBox) {
  const int MaskArea = MaskSize * MaskSize;
  const int DataSize = MaskArea * classNum * sizeof(float);
  auto t_start = std::chrono::high_resolution_clock::now();
  float *pIndex = (float *)keypoints;
  auto stream = nppGetStream();

#if 0  // follow python code: bi-Linear + bi-Cubic, 4x10 keypoints about 3.1ms
  // pIndex = (float *)keypoints;
  float *pMask_host = (float *)g_hbuffer;
  NppiResizeBatchCXR *resizeBatch_dev = (NppiResizeBatchCXR *)(g_dbuffer[2]);
  NppiResizeBatchCXR *resCubicBatch_dev = resizeBatch_dev + classNum;
  for (int i = 0; i < kpMaskNum; i++) {
    Npp32f *pMask_32f = (Npp32f *)dMask + MaskArea * classNum * roiOrder[i];
    Npp32f *pMax = (Npp32f *)(resCubicBatch_dev + classNum) + classNum * 3 * i;
    int *pIndexX = (int *)(pMax + classNum);
    int *pIndexY = (int *)(pIndexX + classNum);
    Npp8u *pDeviceBuffer = (Npp8u *)(pIndexY + classNum);

    NppiResizeBatchCXR resizeBatch[classNum];
    int dstMaskSize = MaskSize * 2;
    {  // bi_linear resize 28x28 -> 56x56
      NppiSize oSrcSize = {MaskSize, MaskSize};  // 28x28
      NppiRect oSrcROI = {0, 0, oSrcSize.width, oSrcSize.height};
      int nSrcStep = oSrcSize.width;

      NppiSize oDstSize = {dstMaskSize, dstMaskSize};       // 56x56
      NppiRect oDstROI = {0, 0, dstMaskSize, dstMaskSize};  //
      int nDstStep = oDstSize.width;
      for (int j = 0; j < classNum; j++) {
        resizeBatch[j].pSrc = pMask_32f + MaskArea * j;
        resizeBatch[j].nSrcStep = nSrcStep * sizeof(float);
        resizeBatch[j].pDst =
            (Npp32f *)g_dbuffer[0] + dstMaskSize * dstMaskSize * j;
        resizeBatch[j].nDstStep = nDstStep * sizeof(float);
      }
      CHECK(cudaMemcpyAsync(resizeBatch_dev, resizeBatch,
                            classNum * sizeof(NppiResizeBatchCXR),
                            cudaMemcpyHostToDevice,
                            stream));  // need copy NppiResizeBatchCXR to device
      NppStatus status =
          nppiResizeBatch_32f_C1R(oSrcSize, oSrcROI, oDstSize, oDstROI,
                                  NPPI_INTER_LINEAR, resizeBatch_dev, classNum);
      DPRINTF(3, "nppiResizeBatch_32f_C1R LINEAR=%d\n", (int)status);
    }

    // bi_Cubic resize 56x56 -> boxSize
    float left = detBox[5 * i + 1];
    float top = detBox[5 * i + 2];
    float width = detBox[5 * i + 3] - left;
    float height = detBox[5 * i + 4] - top;
    int width_ceil = ceil(width);
    int height_ceil = ceil(height);
    NppiSize oSrcSize = {dstMaskSize, dstMaskSize};  // 56x56
    NppiRect oSrcROI = {0, 0, oSrcSize.width, oSrcSize.height};
    int nSrcStep = oSrcSize.width * sizeof(float);
    NppiSize oDstSize = {width_ceil, height_ceil};
    NppiRect oDstROI = {0, 0, oDstSize.width, oDstSize.height};
    int nDstStep = oDstSize.width * sizeof(float);

    int imgSize = width_ceil * height_ceil;
    NppiResizeBatchCXR resCubicBatch[classNum];
    for (int j = 0; j < classNum; j++) {
      resCubicBatch[j].pSrc = resizeBatch[j].pDst;
      resCubicBatch[j].nSrcStep = nSrcStep;
      resCubicBatch[j].pDst = (Npp32f *)g_dbuffer[1] + imgSize * j;
      resCubicBatch[j].nDstStep = nDstStep;
    }
    CHECK(cudaMemcpyAsync(resCubicBatch_dev, resCubicBatch,
                          classNum * sizeof(NppiResizeBatchCXR),
                          cudaMemcpyHostToDevice,
                          stream));  // need copy NppiResizeBatchCXR to device
    NppStatus status =
        nppiResizeBatch_32f_C1R(oSrcSize, oSrcROI, oDstSize, oDstROI,
                                NPPI_INTER_CUBIC, resCubicBatch_dev, classNum);
    DPRINTF(3, "nppiResizeBatch (%d,%d)->(%d,%d) CUBIC =%d\n", dstMaskSize,
            dstMaskSize, width_ceil, height_ceil, (int)status);

    for (int j = 0; j < classNum; j++) {
      status = nppiMaxIndx_32f_C1R((const Npp32f *)resCubicBatch[j].pDst,
                                   nDstStep, oDstSize, pDeviceBuffer, pMax + j,
                                   pIndexX + j, pIndexY + j);
    }
  }

  CHECK(cudaMemcpyAsync(pMask_host, (resCubicBatch_dev + classNum),
                        kpMaskNum * classNum * 3 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  for (int i = 0; i < kpMaskNum; i++) {
    float left = detBox[5 * i + 1];
    float top = detBox[5 * i + 2];
    float width = detBox[5 * i + 3] - left;
    float height = detBox[5 * i + 4] - top;
    int width_ceil = ceil(width);
    int height_ceil = ceil(height);
    float width_correction = width / width_ceil;
    float height_correction = height / height_ceil;
    DPRINTF(3, "w=%d, w_c=%f\n", width_ceil, width_correction);
    float *pMax = pMask_host + classNum * 3 * i;
    int *pIndexX = (int *)(pMax + classNum);
    int *pIndexY = (int *)(pIndexX + classNum);
    for (int j = 0; j < classNum; j++) {
      float maxVal = pMax[j];
      float mx = (pIndexX[j] + 0.5f) * width_correction + left;
      float my = (pIndexY[j] + 0.5f) * height_correction + top;
      DPRINTF(3, "maxIdx=(%f,%f), val=%f\n", mx, my, maxVal);
      short *pMaxIdx = (short *)pIndex;
      pMaxIdx[0] = short(mx);
      pMaxIdx[1] = short(my);
      pIndex[1] = maxVal;
      pIndex += 2;
    }
  }
  DPRINTF(2, "Tot time of KeypointPostProcessing Resize = %fms\n",
          duration<float, std::milli>(high_resolution_clock::now() - t_start)
              .count());
  return;
#endif

  for (int i = 0; i < 1; i++) {  // copy heatmap of first roi
    Npp32f *pMask_32f = (Npp32f *)dMask + MaskArea * classNum * roiOrder[i];
    float *pMask_host = (float *)g_hbuffer + MaskArea * classNum * i;
    CHECK(cudaMemcpyAsync(pMask_host, pMask_32f, DataSize,
                          cudaMemcpyDeviceToHost, stream));
  }
  for (int i = 0; i < kpMaskNum; i++) {
    cudaStreamSynchronize(stream);
    if (i < kpMaskNum - 1) {  // Async copy heatmap of next roi
      Npp32f *pMask_32f =
          (Npp32f *)dMask + MaskArea * classNum * roiOrder[i + 1];
      float *pMask_host = (float *)g_hbuffer + MaskArea * classNum * (i + 1);
      CHECK(cudaMemcpyAsync(pMask_host, pMask_32f, DataSize,
                            cudaMemcpyDeviceToHost, stream));
    }
    float left = detBox[(prob_num + 4) * i + 1];
    float top = detBox[(prob_num + 4) * i + 2];
    float width = detBox[(prob_num + 4) * i + 3] - left;
    float height = detBox[(prob_num + 4) * i + 4] - top;
    float wscale = (width) / float(MaskSize);
    float hscale = (height) / float(MaskSize);

    for (int j = 0; j < classNum; j++, pIndex += 2) {
      float *pMask_host = (float *)g_hbuffer + MaskArea * (classNum * i + j);
      float maxValue = pMask_host[0];
      int maxIdx = 0;
      for (int k = 1; k < MaskArea; k++) {
        if (pMask_host[k] > maxValue) {
          maxValue = pMask_host[k];
          maxIdx = k;
        }
      }
      // scale the pIndex to box size, 4x10 keypoints about 0.5ms
      int sy = maxIdx / MaskSize;  // point in src
      int sx = maxIdx % MaskSize;
      DPRINTF(3, "\n sx=%d,sy=%d \n", sx, sy);
      // adjust the pIndex with interpolateCubic
      {
        // Get 16(4x4) points from src
        float fmax[4][4], *pTmpVal = &fmax[0][0];
        int tmpIdx = 0;
        for (int dy = sy - 1; dy <= sy + 2; dy++) {
          int idySrc = 0;
          if (dy > MaskSize - 1)
            idySrc = (MaskSize - 1) * MaskSize;
          else if (dy > 0)
            idySrc = dy * MaskSize;

          for (int dx = sx - 1; dx <= sx + 2; dx++) {
            int idxSrc = 0;
            if (dx > MaskSize - 1)
              idxSrc = MaskSize - 1;
            else if (dx > 0)
              idxSrc = dx;

            pTmpVal[tmpIdx++] = pMask_host[idySrc + idxSrc];
            DPRINTF(3, "%d:%f ", idySrc + idxSrc, pMask_host[idySrc + idxSrc]);
          }
          DPRINTF(3, "\n");
        }

        // Uses Taylor expansion up to avoid bicubic upscaling, from Caffe2.
        float b[2] = {-(fmax[1][2] - fmax[1][0]) / 2,
                      -(fmax[2][1] - fmax[0][1]) / 2};
        float a = (fmax[2][2] - fmax[2][0] - fmax[0][2] + fmax[0][0]);
        float A[2][2] = {{fmax[1][0] - 2 * fmax[1][1] + fmax[1][2], a / 4},
                         {a / 4, fmax[0][1] - 2 * fmax[1][1] + fmax[2][1]}};
        // Set b=-f'(0), A=f''(0), Solve A*x=b
        float divA = (A[0][0] * A[1][1] - A[0][1] * A[1][0]);  // A.determinant
        float dx = 0, dy = 0;
        if (fabs(divA) > 1e-4) {
          dx = (A[1][1] * b[0] - A[0][1] * b[1]) / divA;
          dy = (A[0][0] * b[1] - A[1][0] * b[0]) / divA;
          // clip dx,dy if going out-of-range of 3x3 grid
          const float MAX_DELTA = 1.5f;
          if (fabs(dx) > MAX_DELTA || fabs(dy) > MAX_DELTA) {
            float largerD = std::max(fabs(dx), fabs(dy));
            dx = dx / largerD * MAX_DELTA;
            dy = dy / largerD * MAX_DELTA;
          }
          maxValue -= (b[0] * dx + b[1] * dy);
          maxValue += (dx * dx * A[0][0] + dy * dy * A[1][1] +
                       dx * dy * (A[0][1] + A[1][0])) /
                      2.0f;
        }
        float my = top + hscale * (sy + 0.5f + dy);  // middle point in dst
        float mx = left + wscale * (sx + 0.5f + dx);
        short *pMaxIdx = (short *)pIndex;
        pMaxIdx[0] = int(mx);
        pMaxIdx[1] = int(my);
        pIndex[1] = maxValue;
        DPRINTF(3, "%d Taylor (%f,%f,%f), maxIdx %d[%.1f,%.1f] %f\n", i, divA,
                dx, dy, maxIdx, my, mx, maxValue);
      }
    }
  }

#if 1
  if (4 <= TRT_DEBUGLEVEL) {     // save to PGM
    const int scaleFactor = -5;  // for nppsConvert: out = in * 2^(-scaleFactor)
    for (int i = 0; i < kpMaskNum; i++) {
      Npp32f *pMask_32f = (Npp32f *)dMask + MaskArea * classNum * roiOrder[i];
      Npp8u *pMask_8u = (Npp8u *)g_dbuffer[0] + MaskArea * classNum * i;
      NppStatus status = nppsConvert_32f8u_Sfs(
          pMask_32f, pMask_8u, MaskArea * classNum, NPP_RND_NEAR, scaleFactor);
      DPRINTF(3, "nppsConvert_32f8u_Sfs = %d\n", status);
    }
    NppiSize imgSize = {MaskSize, (MaskSize * classNum * kpMaskNum)};
    Npp8u *pMask_8u = (Npp8u *)g_dbuffer[0];
    char fileName[260] = "Keypoint_all_1x.pgm";

    int dataSize = imgSize.width * imgSize.height;
    cudaStreamSynchronize(nppGetStream());
    CHECK(cudaMemcpy(g_hbuffer, pMask_8u, dataSize, cudaMemcpyDeviceToHost));

    {
      // Npp8u *pDeviceBuffer = (Npp8u *)dMask;
      const int KPArea = imgSize.width * imgSize.width;

      Npp8u *pKP_8u = (Npp8u *)keypoints;
      // Npp8u *Value = (Npp8u *)dMask + 8;
      int BufferSize;
      pIndex = (float *)keypoints;
      CHECK(nppsMinMaxGetBufferSize_8u(KPArea, &BufferSize));
      DPRINTF(3, "nppsMinMaxGetBufferSize_8u =%dB \n", BufferSize);
      for (int i = 0; i < kpMaskNum; i++) {
        for (int j = 0; j < classNum; j++) {
          pKP_8u[int(*pIndex)] = 255;
          pIndex += 2;
          // pMask_8u += KPArea;
          // DPRINTF(1, "%d nppsMinMaxIndx_8u [%d], [%d]\n", i, nIndex[0],
          // nIndex[1]);
        }
        pKP_8u += KPArea * classNum;
      }
    }

    std::ofstream file(fileName, std::ios::out | std::ios::binary);
    file << "P5\n" << imgSize.width << " " << imgSize.height << "\n255\n";
    file.write((char *)g_hbuffer, dataSize);
    file.close();
    std::cout << "Save PPM: " << fileName << std::endl;
  }
#endif

  DPRINTF(2, "Tot time of KeypointPostProcessing = %fms\n",
          duration<float, std::milli>(high_resolution_clock::now() - t_start)
              .count());
}

void MaskPostProcessing(void *dMask, void *dOutput, void *contours,
                        int *roiOrder, nvinfer1::Dims &buffer_dim, int detNum,
                        float *detBox) {
  auto stream = nppGetStream();
  int classNum = buffer_dim.d[0];
  int height = buffer_dim.d[1];
  int width = buffer_dim.d[2];
  // int maskoutSize =
  //     g_MaskRT[0].bufferDims[1].d[1];  // 28 or 56, MaskSize: 28 fixed
  int maskoutSize = width;
  const int MaskArea = height * width;
  const int scaleFactor = -7;  // for nppsConvert: out = in * 2^(-scaleFactor)
  const int threshold = 1 << (-scaleFactor - 1);  // mask_threshold = 0.5
  int maskborder = maskoutSize / MaskSize;        // 1 or 2
  // const int border = 1;  // need 1pixel border for 28x28
  const int dstMaskHeight =
      (height + 2 * maskborder) * 2;  // add border, then resize
  const int dstMaskWidth = (width + 2 * maskborder) * 2;
  auto t_start = std::chrono::high_resolution_clock::now();
  int maskNum = 0;
  cudaMemsetAsync(g_dbuffer[0], 0, 160 * 64 * detNum * 4, stream);
  cudaMemsetAsync(g_dbuffer[1], 0, 160 * 64 * detNum * 4, stream);
  for (int i = 0; i < detNum; i++) {
    int classID = floor(detBox[i * (prob_num + 4)]);  // classID + 0.1f*roiScore
    if (ID_FREESPACE != classID && 1 == prob_num) continue;

    Npp32f *pMask_32f =
        (Npp32f *)dMask + MaskArea * (classNum * roiOrder[i] + classID);
    Npp8u *pMask_8u =
        (Npp8u *)g_dbuffer[0] +
        width * ((height + 2 * maskborder) * (maskNum++) + maskborder);
    NppStatus status = nppsConvert_32f8u_Sfs(pMask_32f, pMask_8u, MaskArea,
                                             NPP_RND_NEAR, scaleFactor);
    DPRINTF(3, "nppsConvert_32f8u_Sfs = %d\n", status);
  }
  {
    NppiSize oSrcSize = {
        width, (height + 2 * maskborder) * maskNum};  // 28x30n or 56x60n
    NppiRect oSrcROI = {0, 0, oSrcSize.width, oSrcSize.height};
    int nSrcStep = oSrcSize.width;
    NppiSize oDstSize = {dstMaskWidth, dstMaskHeight * maskNum};  // 60x60n
    NppiRect oDstROI = {2 * maskborder, 0, width * 2,
                        dstMaskHeight * maskNum};  // 2,0,56,60n
    int nDstStep = oDstSize.width;
    NppStatus status = nppiResize_8u_C1R(
        (Npp8u *)g_dbuffer[0], nSrcStep, oSrcSize, oSrcROI,
        (Npp8u *)g_dbuffer[1], nDstStep, oDstSize, oDstROI, NPPI_INTER_LINEAR);
    DPRINTF(3, "nppiResize_8u_C1R = %d\n", status);

#if 1
    if (4 <= TRT_DEBUGLEVEL) {  // save to PGM
      char fileName[260] = "Mask_all_th_0.25.pgm";
      int dataSize = oDstSize.width * oDstSize.height;
      cudaStreamSynchronize(stream);
      CHECK(
          cudaMemcpy(contours, g_dbuffer[1], dataSize, cudaMemcpyDeviceToHost));
      std::ofstream file(fileName, std::ios::out | std::ios::binary);
      file << "P5\n" << oDstSize.width << " " << oDstSize.height << "\n255\n";
      file.write((char *)contours, dataSize);
      file.close();
      std::cout << "Save PPM: " << fileName << std::endl;
    }
#endif

    // findContours from mask(dOutput) and save result to  contours
    Npp8u *pData = (Npp8u *)g_dbuffer[1];
    status = nppiThreshold_LTValGTVal_8u_C1IR(pData, nDstStep, oDstSize,
                                              threshold, 0, threshold - 1, 1);

    DPRINTF(3, "nppiThreshold_GT_8u_C1IR = %d\n", status);

    unsigned char *cpu_img = (unsigned char *)g_hbuffer;
    cudaMemcpyAsync(cpu_img, pData,
                    oDstSize.width * oDstSize.height * sizeof(unsigned char),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    findContours(cpu_img, dstMaskWidth, dstMaskHeight, maskborder * 2,
                 maskborder * 2, contours);
  }

  DPRINTF(2, "Tot time of Mask-findContours = %fms\n",
          duration<float, std::milli>(high_resolution_clock::now() - t_start)
              .count());  // about 0.3ms per mask
}

/* findContours
  dMask: device(GPU) buffer for Mask
  maskSize: size of Mask ( width & height ) in dMask
  threshold: threshold of Contours pixels
  contours: result of contours
*/
// int findContours( void* dMask, const int maskSize, int threshold, void*
// contours );

void findContours(unsigned char *InputImage, int Width_i, int Height_i,
                  int pad_height, int pad_width, void *contours) {
  // assume the input image has been padded and the input height and width are
  // the padded image

  // the points stored inside BoundaryPoints are the coordniate of unpadded
  // image
  int nImageSize = Width_i * Height_i;

  float *pointData = (float *)contours;

  if (NULL != InputImage) {
    /*int Offset[8][2] = {
     { -1, -1 },       //  +----------+----------+----------+
     { 0, -1 },          //  |          |          |          |
     { 1, -1 },        //  |(x-1,y-1) | (x,y-1)  |(x+1,y-1) |
     { 1, 0 },         //  +----------+----------+----------+
     { 1, 1 },         //  |(x-1,y)   |  (x,y)   |(x+1,y)   |
     { 0, 1 },         //  |          |          |          |
     { -1, 1 },        //  +----------+----------+----------+
     { -1, 0 }         //  |          | (x,y+1)  |(x+1,y+1) |
     };                    //  |(x-1,y+1) |          |          |
     //  +----------+----------+----------+
     */

    int Offset[8] = {(-1) + (-1 * Width_i), 0 + (-1 * Width_i),
                     1 + (-1 * Width_i),    1 + 0,
                     1 + 1 * Width_i,       0 + 1 * Width_i,
                     (-1) + 1 * Width_i,    (-1) + 0};

    const int NEIGHBOR_COUNT = 8;
    int BoundaryPixelCord;
    int BoundaryStartingPixelCord;
    int BacktrackedPixelCord;

    int Loop,
        prev_Loop;  // recording of the current and previous searching direction
    int TmpPixelCord;

    // int assignID = 1;

    int assignID = 2;

    int contour_color = 1;

    int cnt_path =
        0;  // during contour tracing, how many points has been traced

    // start position and current tracking position on output pointer
    int start_pos = 0;
    int current_pos = 0;

    int CurrentBoundaryCheckingPixelCord;
    int PrevBoundaryCheckingPixxelCord;
    int BackTrackedPixelOffset = 0;

    bool bIsBoundaryFound = false;
    bool bIsStartingBoundaryPixelFound = false;
    // int back_tmp_idx;
    int CurrentBackTrackedPixelOffsetInd;

    // getting the starting pixel of boundary
    // to avoid check Idx-1 >= 0, we start from Idx = 1, skip 1 pixel
    for (int Idx = 1; Idx < nImageSize; ++Idx) {
      if (int(InputImage[Idx]) == contour_color &&
          int(InputImage[Idx - 1]) ==
              0) {  // if pixel is unvisited white and its backtrack pixel is
                    // black, then use this pixel as a start point.

        // Assign start pixel and initialization
        BoundaryPixelCord = Idx;

        BoundaryStartingPixelCord = BoundaryPixelCord;

        BacktrackedPixelCord = Idx - 1;
        BackTrackedPixelOffset = BacktrackedPixelCord - BoundaryPixelCord;

        bIsStartingBoundaryPixelFound = true;

        // Start to trace boundary
        CurrentBoundaryCheckingPixelCord = 0;
        PrevBoundaryCheckingPixxelCord = 0;

        prev_Loop = -1;  // intialize the previous searching direction as -1
        cnt_path++;
        while (bIsStartingBoundaryPixelFound) {
          CurrentBackTrackedPixelOffsetInd = -1;
          for (int Ind = 0; Ind < NEIGHBOR_COUNT; ++Ind) {
            if (BackTrackedPixelOffset == Offset[Ind]) {
              CurrentBackTrackedPixelOffsetInd =
                  Ind;  // Finding the bracktracked pixel's offset index
              break;
            }
          }
          // int Loop = 0;

          // Recording of the current and previous searching direction
          Loop = 0;

          // if cannot find the correct 8 types of offsets in backtrack
          if (CurrentBackTrackedPixelOffsetInd == -1) {
            // cout <<"Error finding offset!"<<endl;
            bIsStartingBoundaryPixelFound = false;
            bIsBoundaryFound = false;
            break;
          }

          while (Loop < (NEIGHBOR_COUNT -
                         1))  // && CurrentBackTrackedPixelOffsetInd != -1 )
          {
            if (Loop == 0)  // need to initialize the prev checking point when
                            // start to search for neighbors
            {
              PrevBoundaryCheckingPixxelCord = BacktrackedPixelCord;
            }
            int OffsetIndex =
                (CurrentBackTrackedPixelOffsetInd + 1) % NEIGHBOR_COUNT;

            CurrentBoundaryCheckingPixelCord =
                BoundaryPixelCord + Offset[OffsetIndex];

            int ImageIndex = CurrentBoundaryCheckingPixelCord;

            if (0 !=
                int(InputImage[ImageIndex]))  // finding the next boundary pixel
            {
              //// Check ID of the current pixel /////////

              if (int(InputImage[ImageIndex]) != contour_color &&
                  int(InputImage[ImageIndex]) != assignID) {
                // cout <<"Repeat boundary, skip!"<<endl;
                // if (InputImage[ImageIndex] != 255)
                //{
                // cout<<"Repeated!"<<endl;
                // cout<<(int)InputImage[ImageIndex]<<endl;
                //}
                bIsStartingBoundaryPixelFound = false;
                bIsBoundaryFound = false;
                break;
              }

              // Optimize the number of output points
              // If the previous searching direction is the same as current
              // searching direction, the tracking is on a straight line -- only
              // record the start and end points, ignore points in the mid Put
              // this code here to record BoundaryPixelCord before it moves
              if (prev_Loop != Loop) {
                // Convert the coordinate from padding image back to original
                // image
                TmpPixelCord =
                    BoundaryPixelCord - (pad_width + pad_height * Width_i);
                pointData[current_pos] = TmpPixelCord;
                current_pos++;
              }
              prev_Loop = Loop;
              cnt_path++;  // record the number of points has been traced

              BoundaryPixelCord = CurrentBoundaryCheckingPixelCord;
              BacktrackedPixelCord = PrevBoundaryCheckingPixxelCord;

              BackTrackedPixelOffset = BacktrackedPixelCord - BoundaryPixelCord;

              InputImage[ImageIndex] = assignID;

              break;
            }
            PrevBoundaryCheckingPixxelCord = CurrentBoundaryCheckingPixelCord;
            CurrentBackTrackedPixelOffsetInd += 1;
            Loop++;
          }

          // Chengzhang added loop stop criteria
          if (Loop >=
              NEIGHBOR_COUNT - 1)  // if searched all surroundings of a pixel
                                   // (except backtrack itself) and still cannot
                                   // find the next connection
          {
            // cout <<"Single non-background pixel, skip!"<<endl;
            bIsStartingBoundaryPixelFound = false;
            bIsBoundaryFound = false;
            break;
          }

          // The simple stop criterion
          // cnt_path > 1 is important to avoid the start point failed when
          // moving to the next point (result into boundarypixel=startpoint with
          // size of boundary 1

          if (cnt_path > 1 &&
              BoundaryPixelCord ==
                  BoundaryStartingPixelCord)  // number of points traced > 1, if
                                              // the current pixel = starting
                                              // pixel
          {
            InputImage[BoundaryStartingPixelCord] = assignID;

            assignID++;

            // Since the BoundaryPoints is currently recording the pixel before
            // moving to next, there is no repeative recording of the start
            // point
            // BoundaryPoints.pop_back();
            bIsBoundaryFound = true;
            break;
          }
        }

        if (!bIsBoundaryFound) {  // If there is no connected boundary clear the
                                  // list
          current_pos = start_pos;
          // BoundaryPoints.clear();
        } else {  // Push the boudary of one connected region

          pointData[current_pos] = -1;
          current_pos++;
          start_pos = current_pos;

          bIsBoundaryFound = false;
          bIsStartingBoundaryPixelFound = false;
        }
        cnt_path = 0;  // assign number of points traced to 0
      }
    }
    // pointData[ptr_last] = -10;//mark as the end of all contours
    pointData[current_pos] = -10;
  }
}
