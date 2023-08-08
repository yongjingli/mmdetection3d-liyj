/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * libonnxtrt.so
 */
#include "NvOnnxParserRuntime.h"

#include <fcntl.h>   // For ::open
#include <unistd.h>  // For ::getopt
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "ResizeBilinear.hpp"  // Resize for upsample in Feng's Model
#include "conv_relu.h"         //For mask-net of maskrcnn
#include "onnxtrt.h"

extern int
    post_nms_topN;  // support to change at converting and adjust at inference.
extern int prob_num;  // add by dyg

// Logger for GIE info/warning/errors
class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;
  std::ostream *_ostream;

 public:
  TRT_Logger(Severity verbosity = Severity::kWARNING,
             std::ostream &ostream = std::cout)
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char *msg) override {
    if (severity <= _verbosity) {
      time_t rawtime = std::time(0);
      char buf[256];
      strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", std::gmtime(&rawtime));
      const char *sevstr =
          (severity == Severity::kINTERNAL_ERROR
               ? "    BUG"
               : severity == Severity::kERROR
                     ? "  ERROR"
                     : severity == Severity::kWARNING
                           ? "WARNING"
                           : severity == Severity::kINFO ? "   INFO"
                                                         : "UNKNOWN");
      (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
    }
  }
};

bool debug_builder = false;
bool ifRunProfile = false;
int TIMING_ITERATIONS = 1;
struct Profiler : public nvinfer1::IProfiler {
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char *layerName, float ms) {
    auto record =
        std::find_if(mProfile.begin(), mProfile.end(),
                     [&](const Record &r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(),
             mProfile[i].second / TIMING_ITERATIONS);
      totalTime += mProfile[i].second;
    }
    printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
  }

} gProfiler;

// Refer from tensorrt4.0 trtexec.cpp
#include <cuda_runtime_api.h>
#include <chrono>
#include "NvInfer.h"

using namespace nvinfer1;

#define CHECK(status)                          \
  {                                            \
    if (status != 0) {                         \
      std::cout << "Cuda failure: " << status; \
      abort();                                 \
    }                                          \
  }

#define CUDA_ERROR_CHECK
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

inline void ShowCudaMemInfo() {
  size_t freeMem;
  size_t totalMem;
  static size_t lastFree;
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "CUDA MemInfo total = " << totalMem
            << "Bytes, free = " << freeMem
            << "Bytes, Delta = " << (int64_t)lastFree - (int64_t)freeMem
            << "Bytes" << std::endl;
  lastFree = freeMem;
}

size_t ReadBinFile(std::string filename, char *&databuffer) {
  size_t size{0};
  std::ifstream file(filename, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    databuffer = new char[size];
    assert(databuffer);
    file.read(databuffer, size);
    file.close();
  }
  return size;
}

int g_inputSize = 0, g_outputSize = 0;

std::vector<int64_t> calculateBindingBufferSizes(const ICudaEngine &engine,
                                                 EngineBuffer bufferInfo[],
                                                 int nbBindings,
                                                 int batchSize) {
  std::vector<int64_t> sizes;
  g_inputSize = 0;
  g_outputSize = 0;
  for (int i = 0; i < nbBindings; ++i) {
    bool isInput = engine.bindingIsInput(i);
    const char *name = engine.getBindingName(i);
    Dims dims = engine.getBindingDimensions(i);
    DataType dtype = engine.getBindingDataType(i);

    int64_t bufferSize = volume(dims) * getElementSize(dtype);
    sizes.push_back(bufferSize);
    if (dims.nbDims < 3) dims.d[2] = 0;
    if (dims.nbDims < 2) dims.d[1] = 0;

    bufferInfo[i].nBufferType = (isInput) ? 0 : 1;
    bufferInfo[i].nDataType = (int)dtype;
    bufferInfo[i].nDims = dims.nbDims;
    bufferInfo[i].nMaxBatch = batchSize;
    memcpy(bufferInfo[i].d, dims.d, sizeof(int) * dims.nbDims);

    DPRINTF(1, "Binding[%d] %s%s %dx[%d,%d,%d],%u\n", i,
            (isInput ? "In:" : "Out:"), name, batchSize, dims.d[0], dims.d[1],
            dims.d[2], getElementSize(dtype));

    if (isInput)
      g_inputSize += (bufferSize * batchSize);
    else
      g_outputSize += (bufferSize * batchSize);
  }
  DPRINTF(1, "TRT_INSIZE=%dB \nTRT_OUTSIZE=%dB\n", g_inputSize, g_outputSize);
  char databuf[33];
  snprintf(databuf, sizeof(databuf), "%d", g_inputSize);
  setenv("TRT_INSIZE", databuf, 1);
  snprintf(databuf, sizeof(databuf), "%d", g_outputSize);
  setenv("TRT_OUTSIZE", databuf, 1);

  return sizes;
}

template <typename DType>
void printOutput(int64_t memSize, void *buffer) {
  int eltCount = memSize / sizeof(DType);
  DType *outputs = new DType[eltCount];
  CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

  for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < 64; ++eltIdx) {
    std::cerr << outputs[eltIdx] << "\t, ";
  }
  if (eltCount > 32) {
    std::cerr << " ... ";
    for (int64_t eltIdx = (eltCount - 64); eltIdx < eltCount; ++eltIdx) {
      std::cerr << outputs[eltIdx] << "\t, ";
    }
  }

  std::cerr << std::endl;
  delete[] outputs;
}

class IWorkspace {
 private:
  int status = 0;  //-1, 0: not inited, >=1: inited,
  int ch = 0;
  IRuntime *infer = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;
  TRT_Logger *trt_logger = nullptr;

  // index for buffer, usefull when different device has different buffer order
  std::vector<int> bufferIndex;
  int nbBindings = 0;
  int nMaxBatchsize = 0;
  std::vector<void *> buffers;
  std::vector<int64_t> buffersSizes;  // Bytes
  cudaStream_t stream = nullptr;
  nvinfer1::Dims bufferDims[ONNXTRT_MAX_BUFFERNUM];
  EngineBuffer bufferInfo[ONNXTRT_MAX_BUFFERNUM];

  int frameID = 0;
  std::string savePath = "";  // save input data

 public:
  IWorkspace() { memset(bufferInfo, 0, sizeof(bufferInfo)); };
  ~IWorkspace() { release(); };

  void setStatus(int s) {
    cudaCheckError();
    status = s;
  }
  bool isInited() { return (nullptr != engine && status > 0); }
  int init(int setch, IRuntime *setinfer, ICudaEngine *setengine,
           TRT_Logger *logger);

  void doInference(float *inputData = NULL, float *outData = NULL,
                   int batchSize = 1, int runTest = 1, int inputType = 0,
                   int outType = 0);

  void release();

  int getBufferInfo(EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
    if (NULL == pNum || NULL == pBufferInfo) return ONNXTRT_PARAMETER_ERR;
    if (NULL == pStream)
      cudaStreamSynchronize(stream);
    else
      *pStream = stream;
    *pNum = nbBindings;
    *pBufferInfo = bufferInfo;
    return 0;
  }

  int saveGPUBuf(void *gpuBuf, int64_t size) {
    if (gpuBuf == nullptr || size <= 0) return -1;

    char *cpuBuf = new char[size];
    CHECK(
        cudaMemcpyAsync(cpuBuf, gpuBuf, size, cudaMemcpyDeviceToHost, stream));
    char outfile[260];
    snprintf(outfile, 256, "%s/ch%d_%d.bin", savePath.c_str(), ch, frameID);
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file.write(cpuBuf, size);
    file.close();
    delete cpuBuf;

    DPRINTF(2, "Save input %s size=%ld\n", outfile, size);
    return size;
  }
};
const int MAX_BATCH = ONNXTRT_MAX_ENGINENUM;
const int MAX_CH = ONNXTRT_MAX_ENGINENUM;
static IWorkspace gworkspace[MAX_CH];

int num_classes = 6;  // cfg.MODEL.NUM_CLASSES = 6
int MaskSize = 28;
const int DETECTNUM = 100;
int MaskOutSize[5] = {
    post_nms_topN * 5, post_nms_topN *num_classes * 4,
    post_nms_topN *num_classes, DETECTNUM *(5 * 5 + 1),
    DETECTNUM *MaskSize *MaskSize
        *num_classes};  // rois/bbox_pred/cls_prob / nms rois / masks
int MaskBlockSize = MaskOutSize[0] + MaskOutSize[1] + MaskOutSize[2] +
                    MaskOutSize[3] + MaskOutSize[4];

int IWorkspace::init(int setch, IRuntime *setinfer, ICudaEngine *setengine,
                     TRT_Logger *logger) {
  if (nullptr == setinfer || nullptr == setengine) return -1;

  ch = setch;
  engine = setengine;
  infer = setinfer;
  trt_logger = logger;

  char *val = getenv("TRT_SAVEPATH");
  if (NULL != val) {
    savePath = val;
    DPRINTF(2, "TRT_SAVEPATH = %s", savePath.c_str());
  }

  {
    char databuf[33];
    nbBindings = engine->getNbBindings();
    assert(nbBindings <= ONNXTRT_MAX_BUFFERNUM);
    buffers.resize(nbBindings);
    bufferIndex.resize(nbBindings);
    nMaxBatchsize = std::min(engine->getMaxBatchSize(), MAX_BATCH);

    buffersSizes = calculateBindingBufferSizes(*engine, bufferInfo, nbBindings,
                                               nMaxBatchsize);

    for (int i = 0; i < nbBindings; ++i) {
      CHECK(cudaMalloc(&buffers[i], (buffersSizes[i] * nMaxBatchsize)));
      bufferInfo[i].p = buffers[i];
      bufferInfo[i].nBufferSize = (buffersSizes[i] * nMaxBatchsize);
      bufferDims[i] = engine->getBindingDimensions(i);
      bufferIndex[i] = i;
    }
    if (9 <= nbBindings) {  // more than one input, need fixed order.
      for (int i = 0; i < nbBindings; ++i) {
        const char *nameInorder[13] = {
            "gpu_0/data_",        "gpu_0/im_info_",    "gpu_0/rois_",
            "gpu_0/bbox_pred_",   "gpu_0/cls_prob_",   "gpu_0/fpn_res2_2_",
            "gpu_0/fpn_res3_3_",  "gpu_0/fpn_res4_5_", "gpu_0/fpn_res5_2_",
            "gpu_0/occupy_prob_", "gpu_0/lever_prob_", "gpu_0/lock_prob_",
            "gpu_0/mask_out_"};
        const char *name = engine->getBindingName(i);
        for (int j = 0; j < nbBindings; j++) {
          if (NULL != strstr(name, nameInorder[j])) {
            bufferIndex[j] = i;
            if (4 == j) {  // adjust post_nms_topN from rois_0 befor inference.
              post_nms_topN = bufferDims[i].d[0];
              num_classes = bufferDims[i].d[1];
              DPRINTF(2, "adjust post_nms_topN=%d num_classes=%d\n",
                      post_nms_topN, num_classes);
              // set the evn if undefined
              snprintf(databuf, sizeof(databuf), "%d", post_nms_topN);
              setenv("TRT_POSTNMS", databuf, 0);
              snprintf(databuf, sizeof(databuf), "%d", num_classes);
              setenv("TRT_CLASS", databuf, 0);

              if (12 <= nbBindings) {  // fsd
                prob_num = 4;          // [score, occupy, lever, lock]
              }
              snprintf(databuf, sizeof(databuf), "%d", prob_num);
              setenv("TRT_PROB_NUM", databuf, 0);

              MaskOutSize[0] = post_nms_topN * 5;
              MaskOutSize[1] = post_nms_topN * num_classes * 4;
              MaskOutSize[2] = post_nms_topN * num_classes * prob_num;
              MaskOutSize[3] = DETECTNUM * ((prob_num + 4) + 4 * 5 + 1);
              MaskOutSize[4] = DETECTNUM * MaskSize * MaskSize * num_classes;
              MaskBlockSize = MaskOutSize[0] + MaskOutSize[1] + MaskOutSize[2] +
                              MaskOutSize[3] + MaskOutSize[4];
            }
          }
        }
      }
    }

    // Mask_RCNN TEST.MAX_SIZE : 1333 or 448
    if (9 <= nbBindings) {
      int idx = bufferIndex[0];
      SetImInfo(bufferDims[idx].d[1], bufferDims[idx].d[2],
                bufferDims[idx].d[1], bufferDims[idx].d[2], 1.0f);
    }

    snprintf(databuf, sizeof(databuf), "%d", bufferDims[bufferIndex[0]].d[1]);
    setenv("TRT_IH", databuf, 0);
    snprintf(databuf, sizeof(databuf), "%d", bufferDims[bufferIndex[0]].d[2]);
    setenv("TRT_IW", databuf, 0);
  }
  DPRINTF(2, "Init nbBindings=%d\n", nbBindings);

  return 0;
}

void IWorkspace::doInference(float *inputData, float *outData, int batchSize,
                             int runTest, int inputType, int outType) {
  // Get the index of input and set the evn of input if undefined
  int inputIndex = bufferIndex[0];
  batchSize = std::min(nMaxBatchsize, batchSize);  // check batchsize

  // DMA the input to the GPU, execute asynchronously, and DMA it back:
  if (inputType >= 0 && nullptr != inputData) {
    int num = buffersSizes[inputIndex] * batchSize;
    if (0 == inputType) {
      buffers[inputIndex] = bufferInfo[inputIndex].p;
      CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, num,
                            cudaMemcpyHostToDevice, stream));
    } else {
      buffers[inputIndex] = inputData;
    }
    DPRINTF(2, "Feed Input[%d] size=%d type=%d\n", inputIndex, num, inputType);
    frameID++;
    if (!savePath.empty()) {
      saveGPUBuf(buffers[inputIndex], num * sizeof(float));
    }
  }

  if (0 == runTest && batchSize > 0) {
    if (nullptr == context) {
      context = engine->createExecutionContext();
      CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, ch));
    }
    if (inputType >= 0) {
      context->enqueue(batchSize, &buffers[0], stream, nullptr);
    }

    if (nullptr != outData) {
      if (0 == outType || -1 == outType) {  // plan copy of all gpu buffer
        int offset = 0;                     // offset of OutData while batch > 1
        for (int ib = 0; ib < nbBindings; ++ib) {
          if (engine->bindingIsInput(ib)) continue;

          int blockSize = buffersSizes[ib] * batchSize;  // Byte
          CHECK(cudaMemcpyAsync(outData + offset, buffers[ib], blockSize,
                                cudaMemcpyDeviceToHost, stream));
          offset += blockSize / 4;
        }
      } else if (1 == outType && 9 <= nbBindings) {  // packed roi & mask
        int offset = 0;  // offset of OutData while batch > 1
        std::vector<void *> curBuffers(buffers);
        nvinfer1::Dims curBufferDims[ONNXTRT_MAX_BUFFERNUM];
        for (int ib = 0; ib < nbBindings; ib++) {
          curBuffers[ib] = buffers[bufferIndex[ib]];
          curBufferDims[ib] = bufferDims[bufferIndex[ib]];
        }
        for (int nb = 0; nb < batchSize; nb++) {  // prcoess batch size <=32
          RunMaskNet(curBuffers, curBufferDims, outData + offset,
                     buffers[inputIndex], stream);
          for (int ib = 0; ib < nbBindings; ib++) {
            curBuffers[ib] += volume(curBufferDims[ib]) * sizeof(float);
          }
          offset += MaskBlockSize;
        }
      } else if (2 == abs(outType)) {  // packed by batchsize
        int offset = 0;                // offset of OutData while batch > 1
        for (int nb = 0; nb < batchSize; ++nb) {
          for (int ib = 0; ib < nbBindings; ++ib) {
            if (engine->bindingIsInput(ib)) continue;

            int blockSize = buffersSizes[ib];  // Byte
            CHECK(cudaMemcpyAsync(outData + offset,
                                  buffers[ib] + nb * blockSize, blockSize,
                                  cudaMemcpyDeviceToHost, stream));
            offset += blockSize / 4;
          }
        }
      } else if (3 == abs(outType)) {  // optimizied for MOD of xpmodel
        int offset = 0;                // offset of OutData while batch > 1
        for (int ib = 0; ib < nbBindings; ++ib) {
          if (engine->bindingIsInput(ib)) continue;

          int blockSize = buffersSizes[ib] * batchSize;  // Byte
          if (2 == bufferDims[ib].nbDims &&
              (34 == bufferDims[ib].d[1] || 36 == bufferDims[ib].d[1])) {
            float th = 0.7f;
            filteMOD(batchSize, buffers[ib], outData + offset, bufferDims[ib],
                     th, buffers[0], buffersSizes[0], stream);
          } else {
            CHECK(cudaMemcpyAsync(outData + offset, buffers[ib], blockSize,
                                  cudaMemcpyDeviceToHost, stream));
          }
          offset += blockSize / 4;
        }
      }
    }

    if (outType >= 0) cudaStreamSynchronize(stream);
    return;  // release stream & buffes in DestoryEngine()
  }

  if (runTest > 0) {  // else: test time only
    int iterations = 3;
    int avgRuns = runTest;
    std::vector<float> times(avgRuns);

    for (int j = 0; j < iterations; j++) {
      if (ifRunProfile && j == 1) {  // skip the first iteration
        context->setProfiler(&gProfiler);
        TIMING_ITERATIONS = (iterations - 1) * avgRuns;
      }

      float total = 0, ms;
      for (int i = 0; i < avgRuns; i++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        context->execute(batchSize, &buffers[0]);

        if (1 == outType && 9 <= nbBindings && nullptr != outData) {
          int offset = 0;  // offset of OutData while batch > 1
          std::vector<void *> curBuffers(buffers);
          nvinfer1::Dims curBufferDims[ONNXTRT_MAX_BUFFERNUM];
          for (int ib = 0; ib < nbBindings; ib++) {
            curBuffers[ib] = buffers[bufferIndex[ib]];
            curBufferDims[ib] = bufferDims[bufferIndex[ib]];
          }
          for (int nb = 0; nb < batchSize; nb++) {  // prcoess batch size <=32
            RunMaskNet(curBuffers, curBufferDims, outData + offset,
                       buffers[inputIndex], stream);
            for (int ib = 0; ib < nbBindings; ib++) {
              curBuffers[ib] += volume(curBufferDims[ib]) * sizeof(float);
            }
            offset += MaskBlockSize;
          }
        } else if (3 == outType) {  // optimizied for MOD of xpmodel
          int offset = 0;           // offset of OutData while batch > 1
          for (int ib = 0; ib < nbBindings; ++ib) {
            if (engine->bindingIsInput(ib)) continue;

            int blockSize = buffersSizes[ib] * batchSize;  // Byte
            if (2 == bufferDims[ib].nbDims &&
                (32 <= bufferDims[ib].d[1] && 40 >= bufferDims[ib].d[1])) {
              float th = 0.7f;
              filteMOD(batchSize, buffers[ib], outData + offset, bufferDims[ib],
                       th, buffers[0], buffersSizes[0], stream);
            } else {
              CHECK(cudaMemcpyAsync(outData + offset, buffers[ib], blockSize,
                                    cudaMemcpyDeviceToHost, stream));
            }
            offset += blockSize / 4;
          }
        }

        if (debug_builder && nullptr != outData) {
          int offset = 0;
          for (int ib = 0; ib < nbBindings; ++ib) {
            int bindingIdx = bufferIndex[ib];
            if (engine->bindingIsInput(bufferIndex[bindingIdx])) continue;

            int blockSize = buffersSizes[bindingIdx] * batchSize;  // Byte
            CHECK(cudaMemcpyAsync(outData + offset, buffers[bindingIdx],
                                  blockSize, cudaMemcpyDeviceToHost, stream));
            offset += blockSize / 4;
          }
        }

        cudaStreamSynchronize(stream);

        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        times[i] = ms;
        total += ms;
      }
      total /= avgRuns;
      std::cout << "CH." << ch << " Average over " << avgRuns << " runs is "
                << total << " ms." << std::endl;

      ShowCudaMemInfo();
    }

    if (debug_builder) {
      for (int ib = 0; ib < nbBindings; ++ib) {
        int bindingIdx = bufferIndex[ib];
        for (int nb = 0; nb < batchSize; ++nb) {
          int64_t bufferSizesOutput = buffersSizes[bindingIdx];
          const char *sBufferType[2] = {"In:", "Out:"};
          DPRINTF(1, "%s[%d].%d buffeSize:%ld Data:\n",
                  sBufferType[bufferInfo[bindingIdx].nBufferType], bindingIdx,
                  nb, bufferSizesOutput);
          printOutput<float>(bufferSizesOutput,
                             buffers[bindingIdx] + nb * bufferSizesOutput);
        }
      }
    }

    if (ifRunProfile && TIMING_ITERATIONS > 0) {
      gProfiler.printLayerTimes();
    }

    return;
  }
}

void IWorkspace::release() {
  status = -2;
  // Release the context, stream and the buffers
  if (nullptr != context) {
    cudaStreamDestroy(stream);
    context->destroy();
    context = nullptr;
  }
  if (nbBindings > 0) {
    for (int i = 0; i < nbBindings; ++i) {
      CHECK(cudaFree(bufferInfo[i].p));
      bufferInfo[i].p = nullptr;
    }
    nbBindings = 0;
    buffers.clear();
  }
  if (nullptr != engine) {
    engine->destroy();
    engine = nullptr;
    infer->destroy();
    infer = nullptr;
  }
  if (nullptr != trt_logger) {
    delete trt_logger;
    trt_logger = nullptr;
  }
  status = 0;
}

// interface for so/python
/*
 * ch: channel ID for multiple models:  0:maskrcnn, 1:resnet ...
 * engine_filename: tensorrt engine file
 * pMaskWeight: mask weight for maskrcnn/retinamask
 */
extern "C" int CreateEngine(int ch, const char *engine_filename,
                            const char *pMaskWeight) {
  if (ch >= 0 && ch < ONNXTRT_MAX_ENGINENUM) {
    if (gworkspace[ch].isInited()) return -2;
  } else if (-1 == ch) {  //-1: Find available ID
    for (int i = ONNXTRT_MAX_ENGINENUM - 1; i >= 0; i--) {
      if (!gworkspace[i].isInited()) {
        ch = i;
        break;
      }
    }
    if (-1 == ch) return -2;
  } else
    return ONNXTRT_PARAMETER_ERR;

  char *trtModelStream{nullptr};
  size_t size = ReadBinFile(engine_filename, trtModelStream);
  if (0 == size) {
    printf("Cannot open %s \n", engine_filename);
    return ONNXTRT_IO_ERR;
  }

  TRT_Logger *trt_logger =
      new TRT_Logger(nvinfer1::ILogger::Severity::kWARNING);
  IRuntime *infer = createInferRuntime(*trt_logger);
  infer->setDLACore(ch % 2);
  ICudaEngine *engine = infer->deserializeCudaEngine(
      trtModelStream, size, nvonnxparser::createPluginFactory(*trt_logger));

  if (trtModelStream) delete[] trtModelStream;
  if (NULL == engine) {
    printf("DeserializeCudaEngine failed from %s \n", engine_filename);
    return ONNXTRT_IO_ERR;
  }
  infer->setDLACore(ch % 2);
  DPRINTF(1, "[CH%d] infer->getDLACore=%d\n", ch, infer->getDLACore());
  gworkspace[ch].init(ch, infer, engine, trt_logger);
  if (NULL != pMaskWeight)  // MaxBatchsize <= 32
  {
    // size of roi_feat for mask: 256*14*14* 4 =200704 Byte, batch<=32
    InitMaskNet((void *)pMaskWeight, 200704 * 32, 32, num_classes);
  }
  gworkspace[ch].setStatus(1);
  return ch;
}

// inputType: 0:cpu float32, 1:gpu float32, 2: xxx
// outType: 0:cpu roi & feat map, 1: cpu roi & mask
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType,
                         char *outData, int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }

  gworkspace[ch].setStatus(-2);
  gworkspace[ch].doInference((float *)inputData, (float *)outData, batch, 0,
                             inputType, outType);
  gworkspace[ch].setStatus(1);
  return ONNXTRT_OK;
}

extern "C" int DestoryEngine(int ch) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  gworkspace[ch].release();

  gworkspace[ch].setStatus(-2);
  DestroyMaskNet();
  gworkspace[ch].setStatus(0);

  return ONNXTRT_OK;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **pBufferInfo, int *pNum,
                                 void **pStream) {
  return gworkspace[ch].getBufferInfo(pBufferInfo, pNum, pStream);
}

int RunProfile(int ch, int batch, char *inputData, int inputType, char *outData,
               int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }

  gworkspace[ch].setStatus(-2);
  gworkspace[ch].doInference((float *)inputData, (float *)outData, batch, 5,
                             inputType, outType);
  gworkspace[ch].setStatus(1);
  return ONNXTRT_OK;
}

// parse mask&keypoint data of Parking model(maskrcnn)
typedef struct {
  int classID;     // id of class [1,class_num]
  float roiScore;  // score of roi [0,1]
  float x;         // rect & mask in scaled image(not the orign image)
  float y;
  float width;
  float height;
  float *pMask;            // raw data of mask
  std::vector<char> mask;  // mask fit the rect, 0: no mask
} TRoiMaskOut;

void ParseMaskData(float *pOutData) {
  float *tmpData = pOutData;

  int roiNum_nms = 0;
  std::vector<TRoiMaskOut> roiMasks;
  for (int outid = 0; outid < 5; outid++) {
    DPRINTF(2, "outData%d[%d] = ", outid, MaskOutSize[outid]);
    for (int eltIdx = 0; eltIdx < MaskOutSize[outid] && eltIdx < 8; ++eltIdx) {
      DPRINTF(2, "%f, ", tmpData[eltIdx]);
    }
    DPRINTF(2, "\n");

    if (3 == outid) {  // nms rois
      for (int i = 0; i < 100; i++) {
        if (tmpData[(prob_num + 4) * i] > 1e-5) {
          int classID =
              floor(tmpData[(prob_num + 4) * i]);  // classID + 0.1f*roiScore
          float roiScore = (tmpData[(prob_num + 4) * i] - classID) * 10;
          float left = tmpData[(prob_num + 4) * i + 1];
          float top = tmpData[(prob_num + 4) * i + 2];
          float right = tmpData[(prob_num + 4) * i + 3];
          float bottom = tmpData[(prob_num + 4) * i + 4];
          // add by dyg
          float occupy = 0, lever = 0, lock = 0;
          if (prob_num == 4) {
            occupy = tmpData[(prob_num + 4) * i + 5];
            lever = tmpData[(prob_num + 4) * i + 6];
            lock = tmpData[(prob_num + 4) * i + 7];
            DPRINTF(2, "occupy=%f lever=%f lock=%f\n", occupy, lever, lock);
          }
          DPRINTF(
              2,
              "%d classID=%d roiScore=%f left=%f top=%f right=%f bottom=%f\n",
              i, classID, roiScore, left, top, right, bottom);
          roiNum_nms = i + 1;

          TRoiMaskOut tmpRoi = {classID,      roiScore,     left, top,
                                right - left, bottom - top, NULL};
          roiMasks.push_back(tmpRoi);
        } else {
          break;
        }
      }
    }

    if (4 == outid) {                // masks
      const int maskSize = 28 * 28;  // float(4byte)
      for (int i = 0; i < roiNum_nms; i++) {
        roiMasks[i].pMask = tmpData + i * maskSize * num_classes;

        float *pData = roiMasks[i].pMask + maskSize * roiMasks[i].classID;
        float *pDataEnd = roiMasks[i].pMask + maskSize;
        DPRINTF(2, "%d classID=%d  Mask={%f, %f, %f, ... %f, %f, %f }\n", i,
                roiMasks[i].classID, pData[0], pData[1], pData[2], pDataEnd[-3],
                pDataEnd[-2], pDataEnd[-1]);
      }
    }

    tmpData += MaskOutSize[outid];
  }

  float im_scale = 1.0f;
  int im_w = 960, im_h = 608;
  // check for Width & Height
  char *val = getenv("TRT_IW");
  if (NULL != val) {
    im_w = atoi(val);
    DPRINTF(1, "getenv TRT_IW=%d\n", im_w);
  }
  val = getenv("TRT_IH");
  if (NULL != val) {
    im_h = atoi(val);
    DPRINTF(1, "getenv TRT_IH=%d\n", im_h);
  }
  int kpNum = 10;
  val = getenv("TRT_KPNUM");
  if (NULL != val) {
    kpNum = atoi(val);
    DPRINTF(1, "getenv TRT_KPNUM=%d\n", kpNum);
  }
  if (kpNum > 10) kpNum = 8;

  for (unsigned int i = 0; i < roiMasks.size(); i++) {
    float ox = roiMasks[i].x * im_scale;
    float oy = roiMasks[i].y * im_scale;
    float w = roiMasks[i].width * im_scale;
    float h = roiMasks[i].height * im_scale;
    // float *pMask = roiMasks[i].pMask;
    int classID = roiMasks[i].classID;
    DPRINTF(1, "Draw classID=%d  rect={%f,%f,%f,%f}\n", classID, ox, oy, w, h);

    float scale_x = w / MaskSize;
    float scale_y = h / MaskSize;
    int tx = -1, ty = -1;  // pre point
    // int tx0 = 0, ty0 = 0;  // first point
    // Draw Mask
    if (classID == 5) {
      float *pContour = roiMasks[0].pMask;
      for (int j = 0; j < (MaskSize * MaskSize); j++) {
        int val = (int)pContour[j];
        if (0 == val || -10 == val) break;

        if (val != -1) {
          int iy = (val / 60 - 0.5f) * scale_y / 2 + oy - 0.5f;
          int ix = (val % 60 - 0.5f) * scale_x / 2 + ox - 0.5f;
          iy = std::min(im_h - 1, std::max(0, iy));
          ix = std::min(im_w - 1, std::max(0, ix));

          if (tx >= 0) {
            DPRINTF(1, " -- %d(%d,%d)", val, iy, ix);
            // DrawLine
          } else {
            DPRINTF(1, "%u Contour %d(%d,%d)", i, val, iy, ix);
            // tx0 = ix;
            // ty0 = iy;
          }
          tx = ix;
          ty = iy;
        } else {
          // DrawLine
          tx = -1;
          ty = -1;
        }
      }
      DPRINTF(1, " ,EndContour\n");
      continue;
    }
    // Draw Keypoint
    if (classID != 5) {
      float *pKP = roiMasks[1].pMask + i * kpNum * 2;  // 10x index&conf
      for (int j = 0; j < kpNum; j++) {
        float conf = pKP[j * 2 + 1];
        short *pIndex = (short *)(pKP + j * 2);
        int ix = pIndex[0];
        int iy = pIndex[1];
        // iy = std::min(im_h - 1, std::max(0, iy));
        // ix = std::min(im_w - 1, std::max(0, ix));
        DPRINTF(1, "%u maxIdx [%d,%d] conf=%f\n", i, iy, ix, conf);
      }
      continue;
    }
  }
}

extern "C" int ParseEngineData(void *input, void *output, int batchSize,
                               int outType) {
  if (1 == outType) {  // maskrcnn
    for (int nb = 0; nb < batchSize; nb++)
      ParseMaskData((float *)input + nb * MaskBlockSize);
    return ONNXTRT_OK;
  }
  return -1;
}

// static funtion called when using dlopen & dlclose
static __attribute__((constructor)) void lib_init(void) {
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
    }
  }
  DPRINTF(1, "Load onnx2trt lib %s built@%s %s DebugLevel=%d\n",
          ONNXTRT_VERSION_STRING, __DATE__, __TIME__, TRT_DEBUGLEVEL);
}

static __attribute__((destructor)) void lib_deinit(void) {
  DPRINTF(1, "Unload onnx2trt lib %s built@%s %s\n", ONNXTRT_VERSION_STRING,
          __DATE__, __TIME__);
}
