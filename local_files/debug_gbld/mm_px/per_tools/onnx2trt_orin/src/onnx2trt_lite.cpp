/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * libonnxtrt_lite.so : simple version without any plugin\profile addin.
 * 2019-5-16 : remove any c++ library like vecotr/fstream/cout.
 */
// Refer from tensorrt4.0 trtexec.cpp
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "NvInfer.h"

using namespace nvinfer1;

int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

#define CHECK(status)                     \
  {                                       \
    if (status != 0) {                    \
      printf("Cuda failure: %d", status); \
      abort();                            \
    }                                     \
  }

// Logger for GIE info/warning/errors
class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;

 public:
  TRT_Logger(Severity verbosity = Severity::kWARNING) : _verbosity(verbosity) {}
  void log(Severity severity, const char *msg) override {
    if (severity <= _verbosity) {
      time_t rawtime = time(0);
      char buf[256];
      strftime(&buf[0], 256, "%Y-%m-%d %H:%M:%S", gmtime(&rawtime));
      const char *sevstr =
          (severity == Severity::kINTERNAL_ERROR
               ? "    BUG"
               : severity == Severity::kERROR
                     ? "  ERROR"
                     : severity == Severity::kWARNING
                           ? "WARNING"
                           : severity == Severity::kINFO ? "   INFO"
                                                         : "UNKNOWN");
      printf("[%s %s] %s\n", buf, sevstr, msg);
    }
  }
};

inline int64_t volume(const nvinfer1::Dims &d) {
  int64_t size = d.d[0];
  for (int i = 1; i < d.nbDims; i++) size *= d.d[i];

  return size;
}

static int g_inputSize = 0, g_outputSize = 0;
int calculateBindingBufferSizes(const ICudaEngine &engine, int nbBindings,
                                int batchSize, int64_t *sizes) {
  g_inputSize = 0;
  g_outputSize = 0;
  for (int i = 0; i < nbBindings; ++i) {
    bool isInput = engine.bindingIsInput(i);
    // const char *name = engine.getBindingName(i);
    Dims dims = engine.getBindingDimensions(i);

    int64_t eltCount = volume(dims) * batchSize;
    sizes[i] = eltCount;

    if (isInput)
      g_inputSize += (eltCount * sizeof(float));
    else
      g_outputSize += (eltCount * sizeof(float));
  }
  DPRINTF(1, "TRT_INSIZE=%d B\nTRT_OUTSIZE=%d B\n", g_inputSize, g_outputSize);
  char databuf[33];
  snprintf(databuf, sizeof(databuf), "%d", g_inputSize);
  setenv("TRT_INSIZE", databuf, 0);
  snprintf(databuf, sizeof(databuf), "%d", g_outputSize);
  setenv("TRT_OUTSIZE", databuf, 0);

  return 0;
}

const int MAX_CH = 16;
const int MAX_BIND = 32;
typedef struct {
  IExecutionContext *context;
  cudaStream_t stream;
  int nbBindings;  // must < MAX_BIND
  void *buffers[MAX_BIND];
  int64_t buffersSizes[MAX_BIND];
  int bufferIndex[MAX_BIND];  // index for buffer,
  nvinfer1::Dims bufferDims[MAX_BIND];
} IWorkspace;

IWorkspace gworkspace[MAX_CH];

void doInference(int ch, ICudaEngine &engine, float *inputData = NULL,
                 float *outData = NULL, int batchSize = 1, int runTest = 1,
                 int inputType = 0, int outType = 0) {
  IExecutionContext *&context = gworkspace[ch].context;
  // Input and output buffer pointers that we pass to the engine - the engine
  // requires exactly IEngine::getNbBindings(), of these, but in this case we
  // know that there is exactly 1 input and 2 output.

  int &nbBindings = gworkspace[ch].nbBindings;
  void **buffers = gworkspace[ch].buffers;
  int64_t *buffersSizes = gworkspace[ch].buffersSizes;
  cudaStream_t &stream = gworkspace[ch].stream;
  nvinfer1::Dims *bufferDims = gworkspace[ch].bufferDims;
  int *bufferIndex = gworkspace[ch].bufferIndex;

  DPRINTF(2, "nbBindings=%d\n", nbBindings);

  if (0 == nbBindings && batchSize > 0) {
    context = engine.createExecutionContext();
    int bindNumber = engine.getNbBindings();
    if (bindNumber > MAX_BIND) return;  // check nbBindings <= MAX_BIND;

    nbBindings = bindNumber;
    calculateBindingBufferSizes(engine, nbBindings, batchSize, buffersSizes);

    for (int i = 0; i < nbBindings; ++i) {
      auto bufferSizesOutput = buffersSizes[i];
      CHECK(cudaMalloc(&buffers[i], bufferSizesOutput * sizeof(float)));
      bufferDims[i] = engine.getBindingDimensions(i);
      gworkspace[ch].bufferIndex[i] = i;
    }

    CHECK(cudaStreamCreate(&stream));
  }

  // Get the index of input and set the evn of input if undefined
  int inputIndex = bufferIndex[0];
  {
    char databuf[33];
    snprintf(databuf, sizeof(databuf), "%d", bufferDims[inputIndex].d[1]);
    setenv("TRT_IH", databuf, 0);
    snprintf(databuf, sizeof(databuf), "%d", bufferDims[inputIndex].d[2]);
    setenv("TRT_IW", databuf, 0);
  }

  // DMA the input to the GPU, execute asynchronously, and DMA it back:
  if (nullptr != inputData) {
    int64_t num = buffersSizes[inputIndex];
    CHECK(cudaMemcpyAsync(
        buffers[inputIndex], inputData, num * sizeof(float),
        (0 == inputType) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice,
        stream));
    DPRINTF(2, "Feed Input size = %ld type=%d \n", num, inputType);
  }

  if (0 == runTest && batchSize > 0) {
    context->execute(batchSize, &buffers[0]);

    if (nullptr != outData) {
      int offset = 0;  // offset of OutData while batch > 1
      for (int ib = 0; ib < nbBindings; ++ib) {
        if (engine.bindingIsInput(ib)) continue;

        int64_t blocknum = buffersSizes[ib];
        CHECK(cudaMemcpyAsync(outData + offset, buffers[ib],
                              blocknum * sizeof(float), cudaMemcpyDeviceToHost,
                              stream));
        offset += blocknum;
      }
    }

    cudaStreamSynchronize(stream);
    return;  // release stream & buffes in DestoryEngine()
  }

  if (0 == batchSize && nbBindings > 0) {
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int i = 0; i < nbBindings; ++i) {
      CHECK(cudaFree(buffers[i]));
    }
    nbBindings = 0;
  }
}

long ReadBinFile(const char *filename, char *&databuffer) {
  long size{0};
  FILE *file = fopen(filename, "rb");
  if (file != NULL) {
    fseek(file, 0, SEEK_END);
    size = (long)ftell(file);
    databuffer = (char *)malloc(size);
    if (NULL == databuffer) {
      fclose(file);
      return -1;
    }
    fseek(file, 0, SEEK_SET);
    size = fread(databuffer, 1, size, file);
    fclose(file);
  }
  return size;
}

// interface for so/python
IRuntime *g_infer[MAX_CH] = {nullptr};
ICudaEngine *g_engine[MAX_CH] = {nullptr};
extern "C" int CreateEngine(int ch, const char *engine_filename,
                            const char *pMaskWeight) {
  DPRINTF(3, "Enter CreateEngine \n");
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
    }
  }

  char *trtModelStream{nullptr};
  long size = ReadBinFile(engine_filename, trtModelStream);
  if (size <= 0) {
    DPRINTF(1, "Cannot open %s \n", engine_filename);
    return -1;
  }

  TRT_Logger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
  g_infer[ch] = createInferRuntime(trt_logger);
  g_engine[ch] =
      g_infer[ch]->deserializeCudaEngine(trtModelStream, size, nullptr);

  if (trtModelStream) {
    free(trtModelStream);
  }
  if (NULL == g_engine[ch]) {
    DPRINTF(1, "DeserializeCudaEngine failed from %s \n", engine_filename);
    return -1;
  }
  doInference(ch, *(g_engine[ch]), nullptr, nullptr, 1, 0, 0, 0);

  return 0;
}

// inputType: 0:cpu float32, 1:gpu float32, 2: xxx
// outType: 0:cpu roi & feat map, 1: cpu roi & mask
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType,
                         char *outData, int outType) {
  DPRINTF(3, "Enter RunEngine \n");
  if (nullptr == g_engine[ch]) {
    return -1;
  }

  doInference(ch, *(g_engine[ch]), (float *)inputData, (float *)outData, batch,
              0, inputType, outType);
  return 0;
}

extern "C" int DestoryEngine(int ch) {
  DPRINTF(3, "Enter DestoryEngine \n");
  if (nullptr == g_engine[ch]) {
    return -1;
  }

  doInference(ch, *(g_engine[ch]), nullptr, nullptr, 0, 0);

  g_engine[ch]->destroy();
  g_engine[ch] = nullptr;
  g_infer[ch]->destroy();
  g_infer[ch] = nullptr;

  return 0;
}
