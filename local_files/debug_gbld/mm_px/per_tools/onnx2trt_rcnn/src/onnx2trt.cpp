/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * bin/onnx2trt -o model_300_320x576b2.trt -b 2 -d 16  -i
 * gpu_0/data_134544.645_320x576.bin -p
 */
#include <fcntl.h>   // For ::open
#include <unistd.h>  // For ::getopt
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include "NvOnnxParserRuntime.h"
#include "ResizeBilinear.hpp"
#include "onnxtrt.h"

// inner defines from onnx2trt.lib
extern int g_inputSize, g_outputSize;
extern int MaskBlockSize;
extern int post_nms_topN;
extern int MaskSize;
extern int num_classes;
extern bool debug_builder;
extern bool ifRunProfile;
size_t ReadBinFile(std::string filename, char *&databuffer);
int RunProfile(int ch, int b, char *i, int inputType, char *o, int outType);

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

inline void ShowCudaMemInfo() {
  size_t freeMem;
  size_t totalMem;
  static size_t lastFree;
  cudaMemGetInfo(&freeMem, &totalMem);
  DPRINTF(1, "CUDA MemInfo total= %zuBytes, free= %zuBytes, Delta= %zuBytes\n",
          totalMem, freeMem,
          (lastFree == 0) ? 0 : ((int64_t)lastFree - freeMem));
  lastFree = freeMem;
}

// code for onnx2trt tool
#include "NvOnnxParser.h"
#include "cuda_profiler_api.h"
#include "onnx_utils.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <thread>
using std::cerr;
using std::cout;
using std::endl;

struct InferDeleter {
  template <typename T>
  void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};
template <typename T>
inline std::shared_ptr<T> infer_object(T *obj) {
  if (!obj) {
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj, InferDeleter());
}

bool ParseFromFile_WAR(google::protobuf::Message *msg, const char *filename) {
  int fd = ::open(filename, O_RDONLY);
  if (fd < 0) {
    cout << "Can't open:" << filename << endl;
    return false;
  }
  google::protobuf::io::FileInputStream raw_input(fd);
  raw_input.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&raw_input);
  // Note: This WARs the very low default size limit (64MB)
  coded_input.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::max() / 4);
  return msg->ParseFromCodedStream(&coded_input);
}

bool ParseFromTextFile(google::protobuf::Message *msg, const char *filename) {
  int fd = ::open(filename, O_RDONLY);
  if (fd < 0) {
    cout << "Can't open:" << filename << endl;
    return false;
  }
  google::protobuf::io::FileInputStream raw_input(fd);
  raw_input.SetCloseOnDelete(true);
  return google::protobuf::TextFormat::Parse(&raw_input, msg);
}

std::string onnx_ir_version_string(
    int64_t ir_version = ::ONNX_NAMESPACE::IR_VERSION) {
  int onnx_ir_major = ir_version / 1000000;
  int onnx_ir_minor = ir_version % 1000000 / 10000;
  int onnx_ir_patch = ir_version % 10000;
  std::ostringstream os;
  os << onnx_ir_major << "." << onnx_ir_minor << "." << onnx_ir_patch;
  return os.str();
}

void print_version() {
  cout << "Parser built against:" << endl;
  cout << "  ONNX IR version:  "
       << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << endl;
  cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "."
       << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << endl;
}

// global params
struct Params {
  std::string deployFile, modelFile, engine,
      calibrationCache{"CalibrationTable"};
  std::string uffFile;
  std::string onnxModelFile;
  int device{0}, batchSize{1}, workspaceSize{16}, iterations{10}, avgRuns{10};
  bool fp16{false}, int8{false}, verbose{false}, hostTime{false};
  float pct{99};
  Dims inputsize;
} gParams;

// print the layers in network
void printLayers(nvinfer1::INetworkDefinition *network,
                 nvinfer1::IBuilder *trt_builder) {
  int layerNum = network->getNbLayers();
  int deviceType = 0;
  double totFlops = 0;
  double totWsize = 0;
#if NV_TENSORRT_MAJOR >= 5
  deviceType = (int)trt_builder->getDefaultDeviceType();
#endif
  for (int i = 0; i < layerNum; i++) {
    auto layer = network->getLayer(i);
#if NV_TENSORRT_MAJOR >= 5
    deviceType = (int)trt_builder->getDeviceType(layer);
#endif
    printf("Layer%d: %s DeviceType=%d\n", i, layer->getName(), deviceType);
    if (layer->getNbInputs() < 1) continue;

    auto type = layer->getType();
    auto inDim = layer->getInput(0)->getDimensions();
    for (int k = 0; k < layer->getNbOutputs(); k++) {
      auto outDim = layer->getOutput(k)->getDimensions();
      long int flops = 0;  // FLOPS
      long int wsize = 0;  // weight
      switch (type) {
        case LayerType::kCONVOLUTION: {
          auto kernelDim = ((IConvolutionLayer *)layer)->getKernelSize();
          flops = (inDim.d[0] * kernelDim.d[0] * kernelDim.d[1] * 2 + 1) *
                  volume(outDim);
          wsize = inDim.d[0] * volume(kernelDim) * outDim.d[0] * 4;
          printf("  FLOPS: %ld (%d,%dx%d,%dx%dx%d) %ld B\n", flops, inDim.d[0],
                 kernelDim.d[0], kernelDim.d[1], outDim.d[0], outDim.d[1],
                 outDim.d[2], wsize);
        } break;
        case LayerType::kFULLY_CONNECTED:
          flops = volume(inDim) * (outDim.d[0] * 3);
          wsize = volume(inDim) * outDim.d[0] * sizeof(float);
          printf("  FLOPS: %ld (%dx%dx%d,%dx%d) %ld B\n", flops, inDim.d[0],
                 inDim.d[1], inDim.d[2], outDim.d[0], outDim.d[1], wsize);
          break;
        case LayerType::kPOOLING: {
          auto kernelDim = ((IPoolingLayer *)layer)->getWindowSize();
          flops = (kernelDim.d[0] * kernelDim.d[1]) * volume(outDim);
          printf("  FLOPS: %ld (%dx%d,%dx%dx%d) \n", flops, kernelDim.d[0],
                 kernelDim.d[1], outDim.d[0], outDim.d[1], outDim.d[2]);
        } break;
        case LayerType::kSCALE:  // batch-norm / scale
          flops = 3 * volume(outDim);
          printf("  FLOPS: %ld (%d,%dx%dx%d) \n", flops, inDim.d[0],
                 outDim.d[0], outDim.d[1], outDim.d[2]);
          break;
        default:
          flops = volume(outDim);  // Sample estimate for
                                   // Relu,ElementWise,Shuffle,Plugin...
      }
      totFlops += flops;
      totWsize += wsize;
    }
  }
  printf("Tot %d Layers, %.3f GFlops, %.3f MB, DefaultDeviceType=%d\n",
         layerNum, totFlops / 1e9, totWsize / 1e6, deviceType);
}

std::vector<int> splitNum(const char *s, char delim, int endMax = -1) {
  std::vector<int> res;
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    res.push_back(atoi(item.c_str()));
  }
  if (endMax > 0 && (res.size() & 1)) res.push_back(endMax);

  return res;
}

float percentile(float percentage, std::vector<float> &times) {
  int all = static_cast<int>(times.size());
  int exclude = static_cast<int>((1 - percentage / 100) * all);
  if (0 <= exclude && exclude <= all) {
    std::sort(times.begin(), times.end());
    float pctTime = times[all == exclude ? 0 : all - 1 - exclude];
    float totTime = 0;
    for (int i = 5; i < all - 5; i++)
      totTime += times[i];  // drop the fist & last 5 datas.

    printf(
        "TestAll %d range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< "
        "%.3fms\n",
        all, times[0], times[all - 1], totTime / (all - 10), times[all / 2],
        percentage, pctTime);
    return pctTime;
  }
  return std::numeric_limits<float>::infinity();
}
// end of Refer

// return the size of image ( C*H*W ) , <0 error
int readPPMFile(const std::string filename, char *&databuffer, int modelType,
                int batchsize = 1) {
  if (0 != filename.compare(filename.size() - 4, 4, ".ppm")) {
    size_t size = ReadBinFile(filename, databuffer);
    printf("ReadBinFile to inputStream size=%zu Bytes\n", size);
    return size;
  }

  std::ifstream infile(filename, std::ifstream::binary);
  if (!infile.good()) return -1;

  char magic[8], w[256], h[8], max[8];
  infile >> magic >> w;
  if (w[0] == '#') {
    infile.getline(w, 255);
    infile >> w;
  }
  infile >> h >> max;
  int imgsize = atoi(w) * atoi(h);
  int size = imgsize;
  if (magic[1] == '6') size *= 3;
  printf("Read magic=%s w=%s, h=%s, s=%dx%d\n", magic, w, h, size, batchsize);
  if (size <= 0 || size > 1e8) return -2;

  float *floatbuffer = new float[size * batchsize];
  unsigned char *rgbbuffer = new unsigned char[size];
  infile.seekg(1, infile.cur);
  infile.read((char *)rgbbuffer, size);
  printf("Readed .tellg=%d\n", (int)infile.tellg());
  int idxRGB[3] = {0, 1, 2};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  if (1 == modelType || 2 == modelType) {  // maskrcnn-benchmark
    Means[0] = 102.9801f;                  // R
    Means[1] = 115.9465f;                  // G
    Means[2] = 122.7717f;                  // B
    idxRGB[2] = 0;                         // Swap R <-> G
    idxRGB[0] = 2;
  }
  if (3 == modelType || 4 == modelType) {  // LLD & DS
    Scales[0] = 255.f;                     // R
    Scales[1] = 255.f;                     // G
    Scales[2] = 255.f;                     // B
  }
  if (7 == modelType) {          // nvidia-retinanet
    Scales[0] = 255.f * 0.229f;  // R
    Scales[1] = 255.f * 0.224f;  // G
    Scales[2] = 255.f * 0.225f;  // B
    Means[0] = 0.485f / 0.229f;  // R
    Means[1] = 0.456f / 0.224f;  // G
    Means[2] = 0.406f / 0.225f;  // B
  }

  for (int i = 0; i < imgsize; i++) {
    floatbuffer[i] = (rgbbuffer[i * 3 + idxRGB[0]] - Means[0]) / Scales[0];
    floatbuffer[i + imgsize] =
        (rgbbuffer[i * 3 + idxRGB[1]] - Means[1]) / Scales[1];  // G
    floatbuffer[i + imgsize * 2] =
        (rgbbuffer[i * 3 + idxRGB[2]] - Means[2]) / Scales[2];  // B
  }
  for (int nb = 1; nb < batchsize; nb++) {
    memcpy(floatbuffer + nb * size, floatbuffer, size * sizeof(float));
  }

  databuffer = (char *)floatbuffer;
  delete[] rgbbuffer;

  return size;
}

// Read images from list for calibrator
class CalibratorFromList {
 public:
  CalibratorFromList(std::string listFile, std::string cacheFile, int mType = 4,
                     int nBatch = 1)
      : mTotalSamples(0), mCurrentSample(0), mCacheFile(cacheFile) {
    modelType = mType;
    batchSize = nBatch;
    mTotalSamples = 1;  // default
    int imgsize = batchSize * volume(gParams.inputsize);
    if (!listFile.empty()) {
      std::ifstream infile(listFile);
      if (!infile.good()) {
        DPRINTF(1, "Int8CalibratorFromList %s faild!\n", listFile.c_str());
        return;
      }

      // Read every line as image path from input file
      std::string imgPath;
      while (std::getline(infile, imgPath)) {
        if (imgPath.length() > 2) mImageList.push_back(imgPath);
      }
      mTotalSamples = mImageList.size();
      DPRINTF(1, "Read %d ImageList from %s\n", mTotalSamples,
              listFile.c_str());
    } else
      DPRINTF(1, "Set const Data for Int8Calibrator\n");

    void *cudaBuffer;
    CHECK(cudaMalloc(&cudaBuffer, imgsize * sizeof(float)));
    CHECK(cudaMemset(cudaBuffer, 0x40, imgsize * sizeof(float)));  // 3.0f
    mInputDeviceBuffers.insert(std::make_pair(gInputs[1], cudaBuffer));
    gInputSize[1] = imgsize;

    // alloc the cuda memory for im_info ( 3 floats )
    CHECK(cudaMalloc(&cudaBuffer, gInputSize[0] * sizeof(float)));
    mInputDeviceBuffers.insert(std::make_pair(gInputs[0], cudaBuffer));
  }

  virtual ~CalibratorFromList() {
    for (auto &elem : mInputDeviceBuffers) CHECK(cudaFree(elem.second));
  }

  int getBatchSize() const { return batchSize; }

  bool getBatch(void *bindings[], const char *names[], int nbBindings) {
    if (mCurrentSample >= mTotalSamples) return false;

    if (0 < nbBindings) {
      int inputIndex = 0;
      if (2 == nbBindings) {
        DPRINTF(1, "getBatch[%d] nbBindings=%d %s %s\n", mCurrentSample,
                nbBindings, names[0], names[1]);
        bindings[0] = mInputDeviceBuffers[names[0]];

        inputIndex = 1;
      } else
        DPRINTF(1, "getBatch%d[%d] nbBindings=%d %s modelType=%d\n", batchSize,
                mCurrentSample, nbBindings, names[0], modelType);

      bindings[inputIndex] = mInputDeviceBuffers[gInputs[1]];
      int blocksize = gInputSize[1] * sizeof(float);
      if (mImageList.size() > 0) {
        const char *input_filename = mImageList[mCurrentSample].c_str();
        char *databuffer;
        int ret = readPPMFile(input_filename, databuffer, modelType);
        if (ret >= gInputSize[1]) {
          CHECK(cudaMemcpy(bindings[inputIndex], databuffer, blocksize,
                           cudaMemcpyHostToDevice));
        } else {
          DPRINTF(1, "Error! readPPMFile %s size=%d, Request=%d. Skiped.\n",
                  input_filename, ret, gInputSize[1]);
        }

        delete[] databuffer;
      }
    }

    ++mCurrentSample;
    return true;
  }

  const void *readCalibrationCache(size_t &length) {
    // return nullptr;
    mCalibrationCache.clear();
    std::ifstream input(mCacheFile, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
      DPRINTF(1, "readCalibrationCache\n");
      std::copy(std::istream_iterator<char>(input),
                std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));
    }
    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
  }

  virtual void writeCalibrationCache(const void *cacheData, size_t dataSize) {
    std::ofstream file(mCacheFile, std::ios::out | std::ios::binary);
    if (file.is_open()) {
      file.write((const char *)cacheData, dataSize);
      file.close();
    }
  }

 private:
  int mTotalSamples;
  int mCurrentSample;
  std::string mCacheFile;
  std::map<std::string, void *> mInputDeviceBuffers;
  std::vector<char> mCalibrationCache;
  std::vector<std::string> mImageList;

  int modelType;
  int batchSize;
  std::vector<std::string> gInputs{"gpu_0/im_info_0", "gpu_0/data_0"};
  std::vector<int> gInputSize{3, 1 * 3 * 302 * 480};
};

class Int8Calibrator : public IInt8EntropyCalibrator {
 public:
  Int8Calibrator(std::string listFile, std::string cacheFile, int mType = 4,
                 int nBatch = 1) {
    calibrator = new CalibratorFromList(listFile, cacheFile, mType, nBatch);
  }

  ~Int8Calibrator() { delete calibrator; }

  int getBatchSize() const override { return calibrator->getBatchSize(); }

  bool getBatch(void *bindings[], const char *names[],
                int nbBindings) override {
    return calibrator->getBatch(bindings, names, nbBindings);
  }

  const void *readCalibrationCache(size_t &length) override {
    return calibrator->readCalibrationCache(length);
  }

  virtual void writeCalibrationCache(const void *cacheData,
                                     size_t dataSize) override {
    return calibrator->writeCalibrationCache(cacheData, dataSize);
  }

 private:
  CalibratorFromList *calibrator;
};

#if NV_TENSORRT_MAJOR >= 6
class DLACalibrator : public IInt8EntropyCalibrator2 {
 public:
  DLACalibrator(std::string listFile, std::string cacheFile, int mType = 4,
                int nBatch = 1) {
    calibrator = new CalibratorFromList(listFile, cacheFile, mType, nBatch);
  }

  ~DLACalibrator() { delete calibrator; }

  int getBatchSize() const override { return calibrator->getBatchSize(); }

  bool getBatch(void *bindings[], const char *names[],
                int nbBindings) override {
    return calibrator->getBatch(bindings, names, nbBindings);
  }

  const void *readCalibrationCache(size_t &length) override {
    return calibrator->readCalibrationCache(length);
  }

  virtual void writeCalibrationCache(const void *cacheData,
                                     size_t dataSize) override {
    return calibrator->writeCalibrationCache(cacheData, dataSize);
  }

 private:
  CalibratorFromList *calibrator;
};
#else  // no IInt8EntropyCalibrator2 in TensorRT5.0
typedef Int8Calibrator DLACalibrator;
#endif

static int testRunEngine(int ch, int batch, char *inputData, int inputType,
                         char *outData, int outType) {
  auto t_start = std::chrono::high_resolution_clock::now();

  int nRet = RunEngine(ch, batch, inputData, inputType, outData, outType);

  float ms = std::chrono::duration<float, std::milli>(  // 0.4us
                 std::chrono::high_resolution_clock::now() - t_start)
                 .count();
  DPRINTF(1, "CH[%d] time = %f\n", ch, ms);  // cout : 0.1ms, printf: 0.03ms

  return nRet;
}

extern "C" void markuseroutput(const char *poutput);
extern "C" void markuserinput(const char *pinput);

void print_usage() {
  cout << "ONNX to TensorRT model parser" << endl;
  cout << "Usage: onnx2trt model.onnx\n"
       << "\t\t[-o engine_file.trt]  (output or test TensorRT "
          "engine)\n"
       << "\t\t[-m modelType|masknet.trt ](modelType=1,2,4)\n"
       << "\t\t[-O outputlayers]  (name of output layer)(multiple)\n"
       << "\t\t[-e engine_file.trt]  (test TensorRT "
          "engines(multiple), can be specified multiple times)\n"
       << "\t\t[-i input_data.bin]  (input data(multiple))\n"
       << "\t\t[-t onnx_model.pbtxt] (output ONNX text file "
          "without weights)\n"
       << "\t\t[-T onnx_model.pbtxt] (output ONNX text file with "
          "weights)\n"
       << "\t\t[-b max_batch_size (default 1)]\n"
       << "\t\t[-w max_workspace_size_bytes (default 16 MiB)]\n"
       << "\t\t[-d model_data_type_bit_depth] (default float32, 16 "
          "=> float16, 8 => int8)\n"
       << "\t\t[-D DLA_layers_Number](support float16/int8)\n"
       << "\t\t[-I Int8 layer index](1,8,16,32 => 1~8 & 16~32)\n"
       << "\t\t[-F FP32 layer index](1,8,16,32 => 1~8 & 16~32)\n"
       << "\t\t[-C imageList] (Cablition Int8 from ppm Image list)\n"
       << "\t\t[-c Cablition Cache] (Cablition cache file)\n"
       << "\t\t[-N post_nms_topN](100~300, TRT_PRE_NMS <= 512)\n"
       << "\t\t[-l] (list layers and their shapes)\n"
       << "\t\t[-g] (debug mode, print outdata as floats)\n"
       << "\t\t[-v] (increase verbosity)\n"
       << "\t\t[-q] (decrease verbosity)\n"
       << "\t\t[-p] (enalbe profiler when run exection)\n"
       << "\t\t[-V] (show version information)\n"
       << "\t\t[-h] (show help)\n"
       << "Environment:\n"
       << "\t\tTRT_DEBUGLEVEL: Control the Debug output, -1~4, default:1\n"
       << "\t\tTRT_IC, TRT_IW, TRT_IH, TRT_ID: Input CHW and Type(8/16/32)\n"
       << "\t\tTRT_GEMV: Enable Int8GEMV and set TopK of weight, 1~2\n"
       << "\t\tTRT_FEAT: Fixed the number of feature map for MaskRCNN, 4~5\n";
}

int main(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  ShowCudaMemInfo();
  std::string engine_filename;
  std::vector<std::string> engine_filenames;
  std::string input_filename;
  std::vector<std::string> input_filenames;
  std::string text_filename;
  std::string full_text_filename;
  size_t max_batch_size = 1;
  size_t calib_batch = 1;               // batch size for INT8 calibrator
  size_t max_workspace_size = 1 << 24;  // default: 16MiB, Parking > 128MB
  int model_dtype_nbits = 32;
  int model_DLA_Layer_Num = 0;        // the layes Num of DLA, [0,TotLayers)
  std::vector<int> model_INT8_Layer;  // the begin-end index of layes for int8
  std::vector<int> model_FP32_Layer;  // the begin-end index of layes for fp32
  std::vector<int> model_FP16_Layer;  // the begin-end index of layes for fp16
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  bool print_layer_info = false;
  int modelType = 4;              // 1: maskrcnn, 4: ldd&ds 2: retinanet
  char *mask_filename = nullptr;  //"fastrcnn_nomaskfile";
  std::string Calibrator_imgList;
  int arg = 0;
  float fps_delay = 30;  // delay ms to get fixed FPS

  {
    int low, high;
    cudaDeviceGetStreamPriorityRange(&low, &high);
    DPRINTF(2, "StreamPriorityRange:[%d,%d]\n", low, high);
    DPRINTF(2, "sizeof cudaStream_t = %ld\n", sizeof(cudaStream_t));
    DPRINTF(2, "sizeof void* = %ld\n", sizeof(void *));
    DPRINTF(2, "sizeof EngineBuffer = %ld\n", sizeof(EngineBuffer));
  }

  const char *opt_string = "o:e:i:b:B:w:t:T:d:D:m:I:S:F:C:c:N:O:lgvqVhp";
  while ((arg = ::getopt(argc, argv, opt_string)) != -1) {
    if ('o' == arg || 'e' == arg || 'i' == arg || 'b' == arg || 'w' == arg ||
        't' == arg || 'T' == arg || 'd' == arg || 'D' == arg || 'm' == arg ||
        'I' == arg || 'S' == arg || 'C' == arg || 'N' == arg || 'O' == arg) {
      if (!optarg) {
        cerr << "ERROR: -" << arg << " flag requires argument" << endl;
        return -1;
      }
    }

    switch (arg) {
      case 'o':
        engine_filename = optarg;
        engine_filenames.push_back(optarg);
        break;
      case 'O':
        markuseroutput(optarg);
        break;
      case 'e':
        engine_filenames.push_back(optarg);
        break;
      case 'i':
        input_filename = optarg;
        input_filenames.push_back(optarg);
        break;
      case 't':
        text_filename = optarg;
        break;
      case 'T':
        full_text_filename = optarg;
        break;
      case 'b':
        max_batch_size = atoll(optarg);
        break;
      case 'B':
        calib_batch = atoll(optarg);
        break;
      case 'w':
        max_workspace_size = atoll(optarg);
        break;
      case 'd':
        model_dtype_nbits = atoi(optarg);
        break;
      case 'D':
        model_DLA_Layer_Num = atoi(optarg);
        break;
      case 'I':  // int8
        model_INT8_Layer = (splitNum(optarg, ',', 1e4));
        break;
      case 'S':  // fp32
        model_FP32_Layer = (splitNum(optarg, ',', 1e4));
        break;
      case 'F':  // fixed FPS = 20;
        fps_delay = 1000.f / (atoi(optarg) + 0.01);
        break;
      case 'H':  // fp16
        model_FP16_Layer = (splitNum(optarg, ',', 1e4));
        break;
      case 'm':
        if (strlen(optarg) == 1) {
          modelType = atoi(optarg);
        } else {
          modelType = 1;
          mask_filename = optarg;
        }
        break;
      case 'C':  //
        Calibrator_imgList = optarg;
        break;
      case 'c':  //
        gParams.calibrationCache = optarg;
        break;
      case 'N':
        post_nms_topN = atoi(optarg);
        break;
      case 'l':
        print_layer_info = true;
        break;
      case 'g':
        debug_builder = true;
        break;
      case 'v':
        ++verbosity;
        TRT_DEBUGLEVEL++;
        break;
      case 'q':
        --verbosity;
        break;
      case 'p':
        ifRunProfile = true;
        break;
      case 'V':
        print_version();
        return 0;
      case 'h':
        print_usage();
        return 0;
    }
  }

  TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
  auto trt_builder = infer_object(nvinfer1::createInferBuilder(trt_logger));
  auto trt_network = infer_object(trt_builder->createNetwork());

  bool fp16 = trt_builder->platformHasFastFp16();
  bool int8 = trt_builder->platformHasFastInt8();
  nvinfer1::DataType model_dtype;
  if (model_dtype_nbits == 32) {
    model_dtype = nvinfer1::DataType::kFLOAT;
  } else if (model_dtype_nbits == 16) {
    model_dtype = nvinfer1::DataType::kHALF;
  } else if (model_dtype_nbits == 8) {
    model_dtype = nvinfer1::DataType::kINT8;
  } else {
    cerr << "ERROR: Invalid model data type bit depth: " << model_dtype_nbits
         << endl;
    return -2;
  }

  char *inputStream{nullptr};
  if (!input_filename.empty()) {
    readPPMFile(input_filename, inputStream, modelType, max_batch_size);
  }

  int num_args = argc - optind;
  int ch_num = engine_filenames.size();
  int outType = (modelType == 1 || modelType == 3) ? modelType : 0;
  if (num_args != 1) {
    if (ch_num == 0) {
      print_usage();
      return 0;
    } else if (!engine_filename.empty()) {
      // load directly from serialized engine file if onnx model not specified
      cudaProfilerStop();
      CreateEngine(0, engine_filename.c_str(), mask_filename);
      RunEngine(0, max_batch_size, inputStream, 0, nullptr, outType);
      cudaProfilerStart();
      {
        char *output = new char[g_outputSize];
        RunProfile(0, max_batch_size, inputStream, 0, output, outType);
        ParseEngineData(output, nullptr, max_batch_size, outType);
        delete[] output;
      }
      cudaProfilerStop();
      DestoryEngine(0);
    } else {  // run models parallel
      cudaProfilerStop();
      char *output[16];
      int chID[16];
      //"gpu_0/masknetweight.bin"
      for (int ch = 0; ch < ch_num; ch++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        chID[ch] =
            CreateEngine(-1, engine_filenames[ch].c_str(), mask_filename);
        float ms = std::chrono::duration<float, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_start)
                       .count();

        printf("[%d]CreateEngine CHID.%d time = %fms\n", ch, chID[ch], ms);
        output[ch] = new char[g_outputSize];
        RunEngine(chID[ch], max_batch_size, inputStream, 0, output[ch],
                  outType);
        mask_filename = NULL;
      }
      cudaProfilerStart();
      const float pct = 90.0f;
      const int MAX_TEST = 100;
      std::vector<float> times(MAX_TEST);
      for (int nT = 0; nT < MAX_TEST; nT++) {
        auto t_start = std::chrono::high_resolution_clock::now();
#if 1  // USE_OMP
       //#pragma omp parallel for
        for (int ch = 0; ch < ch_num; ch++) {
          RunEngine(chID[ch], max_batch_size, inputStream, 0, output[ch], -1);

          float ms = std::chrono::duration<float, std::milli>(
                         std::chrono::high_resolution_clock::now() - t_start)
                         .count();
          printf("%d CH.%d time = %f\n", ch, chID[ch], ms);
        }
        EngineBuffer *pBufferInfo[16];
        int bufNum[16];
        void *stream;
        for (int ch = 0; ch < ch_num; ch++) {
          GetBufferOfEngine(chID[ch], &pBufferInfo[ch], &bufNum[ch], &stream);
        }
        // cudaDeviceSynchronize();
#else
        xpilot_os::Thread t[ch_num];
        for (int ch = 0; ch < ch_num; ch++) {
          t[ch] = xpilot_os::Thread("cp_TestRunEngine", testRunEngine, chID[ch], max_batch_size,
                              inputStream, 0, output[ch], outType);
        }
        for (int ch = 0; ch < ch_num; ch++) {
          t[ch].join();
        }
#endif
        float ms = std::chrono::duration<float, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_start)
                       .count();            // 0.4us
        DPRINTF(0, "Tot time = %f\n", ms);  // cout : 0.1ms, printf: 0.03ms
        times[nT] = ms;
        if (ms < fps_delay) usleep((fps_delay - ms) * 1000);
      }
      cudaProfilerStop();
      percentile(pct, times);

      for (int ch = 0; ch < ch_num; ch++) {
        DestoryEngine(chID[ch]);
        delete[] output[ch];
      }
    }
  } else {
    std::string onnx_filename = argv[optind];

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    bool is_binary = ParseFromFile_WAR(&onnx_model, onnx_filename.c_str());
    if (!is_binary && !ParseFromTextFile(&onnx_model, onnx_filename.c_str())) {
      cerr << "Failed to parse ONNX model" << endl;
      return -3;
    }

    if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
      int64_t opset_version = (onnx_model.opset_import().size()
                                   ? onnx_model.opset_import(0).version()
                                   : 0);
      cout << "----------------------------------------------------------------"
           << endl;
      cout << "Input filename:   " << onnx_filename << endl;
      cout << "ONNX IR version:  "
           << onnx_ir_version_string(onnx_model.ir_version()) << endl;
      cout << "Opset version:    " << opset_version << endl;
      cout << "Producer name:    " << onnx_model.producer_name() << endl;
      cout << "Producer version: " << onnx_model.producer_version() << endl;
      cout << "Domain:           " << onnx_model.domain() << endl;
      cout << "Model version:    " << onnx_model.model_version() << endl;
      cout << "Doc string:       " << onnx_model.doc_string() << endl;
      cout << "----------------------------------------------------------------"
           << endl;
    }

    if (onnx_model.ir_version() > ::ONNX_NAMESPACE::IR_VERSION) {
      cerr << "WARNING: ONNX model has a newer ir_version ("
           << onnx_ir_version_string(onnx_model.ir_version())
           << ") than this parser was built against ("
           << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << ")."
           << endl;
    }

    if (!text_filename.empty()) {
      if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
        cout << "Writing ONNX model (without weights) as text to "
             << text_filename << endl;
      }
      std::ofstream onnx_text_file(text_filename.c_str());
      std::string onnx_text = pretty_print_onnx_to_string(onnx_model);
      onnx_text_file.write(onnx_text.c_str(), onnx_text.size());
    }
    if (!full_text_filename.empty()) {
      if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
        cout << "Writing ONNX model (with weights) as text to "
             << full_text_filename << endl;
      }
      std::string full_onnx_text;
      google::protobuf::TextFormat::PrintToString(onnx_model, &full_onnx_text);
      std::ofstream full_onnx_text_file(full_text_filename.c_str());
      full_onnx_text_file.write(full_onnx_text.c_str(), full_onnx_text.size());
    }

    auto trt_parser =
        infer_object(nvonnxparser::createParser(trt_network.get(), trt_logger));

    if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
      cout << "Parsing model" << endl;
    }

    std::ifstream onnx_file(onnx_filename.c_str(),
                            std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);
    if (!onnx_file.read(onnx_buf.data(), onnx_buf.size())) {
      cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
      return -4;
    }
    if (!trt_parser->parse(onnx_buf.data(), onnx_buf.size())) {
      int nerror = trt_parser->getNbErrors();
      for (int i = 0; i < nerror; ++i) {
        nvonnxparser::IParserError const *error = trt_parser->getError(i);
        if (error->node() != -1) {
          ::ONNX_NAMESPACE::NodeProto const &node =
              onnx_model.graph().node(error->node());
          cerr << "While parsing node number " << error->node() << " ["
               << node.op_type();
          if (node.output().size()) {
            cerr << " -> \"" << node.output(0) << "\"";
          }
          cerr << "]:" << endl;
          if (verbosity >= (int)nvinfer1::ILogger::Severity::kINFO) {
            cerr << "--- Begin node ---" << endl;
            cerr << node << endl;
            cerr << "--- End node ---" << endl;
          }
        }
        cerr << "ERROR: " << error->file() << ":" << error->line()
             << " In function " << error->func() << ":\n"
             << "[" << static_cast<int>(error->code()) << "] " << error->desc()
             << endl;
      }
      return -5;
    }

    if (print_layer_info)  // print the layers in network
    {
      printLayers(trt_network.get(), trt_builder.get());
    }

    if (!engine_filename.empty()) {
      gParams.inputsize = trt_network.get()->getInput(0)->getDimensions();
      if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
        DPRINTF(1, "Platform support FP16:%d, INT8:%d\n", fp16, int8);
        DPRINTF(1, "Building TensorRT engine, \n");
        DPRINTF(1, "\tInputSize: [%d,%d,%d]\n", gParams.inputsize.d[0],
                gParams.inputsize.d[1], gParams.inputsize.d[2]);
        DPRINTF(1, "\tDataType(-d): %d bits\n", model_dtype_nbits);
        DPRINTF(1, "\tMax batch(-b): %zu\n", max_batch_size);
        DPRINTF(1, "\tMax workspace(-w): %zu B\n", max_workspace_size);
      }
      std::unique_ptr<IInt8Calibrator> calibrator;
      trt_builder->setMaxBatchSize(max_batch_size);
      trt_builder->setMaxWorkspaceSize(max_workspace_size);
      if (fp16 && model_dtype == nvinfer1::DataType::kHALF) {
        trt_builder->setHalf2Mode(true);
      } else if (int8 && model_dtype == nvinfer1::DataType::kINT8) {
        if (0 != model_DLA_Layer_Num) {
          calibrator.reset(new DLACalibrator(Calibrator_imgList,
                                             gParams.calibrationCache,
                                             modelType, calib_batch));
        } else
          calibrator.reset(new Int8Calibrator(Calibrator_imgList,
                                              gParams.calibrationCache,
                                              modelType, calib_batch));
        trt_builder->setInt8Mode(true);
        trt_builder->setInt8Calibrator(calibrator.get());
      }
#if NV_TENSORRT_MAJOR >= 5  // TensorRT5 support DLA & setStrictTypeConstraints
      trt_builder->allowGPUFallback(true);
      trt_builder->setDLACore(0);
      if (0 != model_DLA_Layer_Num) {
        trt_builder->setDefaultDeviceType(DeviceType::kDLA);
      }
      if (-1 == model_DLA_Layer_Num) {  // Use safe model, all layer run on DLA
        trt_builder->setEngineCapability(EngineCapability::kSAFE_DLA);
        int tfmt = (model_dtype == nvinfer1::DataType::kINT8) ? 0x20 : 0x10;
        auto input0 = trt_network->getInput(0);
        input0->setType(model_dtype);
        input0->setAllowedFormats(tfmt);
        auto output0 = trt_network->getOutput(0);
        output0->setType(model_dtype);
        output0->setAllowedFormats(tfmt);
      } else
        trt_builder->allowGPUFallback(true);  // Use GPU

      int int8SliceNum = model_INT8_Layer.size();
      int fp32SliceNum = model_FP32_Layer.size();
      if (int8SliceNum > 0) {
        trt_builder->setFp16Mode(true);
      }
      if (int8SliceNum > 0 || fp32SliceNum > 0) {
        trt_builder->setStrictTypeConstraints(true);
      }

      int layerNum = trt_network->getNbLayers();
      for (int i = 0; i < layerNum; i++) {
        auto layer = trt_network->getLayer(i);
        const char *pName = layer->getName();
        if (int8SliceNum > 0 || fp32SliceNum > 0) {
          auto dataType = nvinfer1::DataType::kFLOAT;
          bool isINT8Layer = false;
          bool isFP32Layer = false;
          for (int j = 0; j < int8SliceNum; j += 2) {
            if (i >= model_INT8_Layer[j] && i <= model_INT8_Layer[j + 1])
              isINT8Layer = true;
          }
          for (int j = 0; j < fp32SliceNum; j += 2) {
            if (i >= model_FP32_Layer[j] && i <= model_FP32_Layer[j + 1])
              isFP32Layer = true;
          }
          // force layer to execute with required precision
          if (isFP32Layer) {
            DPRINTF(1, "set kFLOAT at layer[%d]%s\n", i, pName);
            dataType = nvinfer1::DataType::kFLOAT;
          } else if (isINT8Layer) {
            DPRINTF(1, "set kINT8 at layer[%d]%s\n", i, pName);
            dataType = nvinfer1::DataType::kINT8;
          } else {
            DPRINTF(1, "set kHALF at layer[%d]%s\n", i, pName);
            dataType = nvinfer1::DataType::kHALF;
          }
          layer->setPrecision(dataType);
          for (int j = 0; j < layer->getNbOutputs(); ++j) {
            // layer->setOutputType(j, dataType);
          }
        }
        // set the layes befor model_DLA_Layer_Num to run DLA
        if (model_DLA_Layer_Num > 0) {
          if (i <= model_DLA_Layer_Num) {
            trt_builder->setDeviceType(layer, DeviceType::kDLA);
            DPRINTF(1, "setDeviceType kDLA at layer[%d]%s\n", i, pName);
          } else {
            trt_builder->setDeviceType(layer, DeviceType::kGPU);
            DPRINTF(1, "setDeviceType kGPU at layer[%d]%s\n", i, pName);
          }
        }
      }
#endif

      trt_builder->setDebugSync(debug_builder);
      auto trt_engine =
          infer_object(trt_builder->buildCudaEngine(*trt_network.get()));
      calibrator.reset();
      if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
        cout << "Writing TensorRT engine to " << engine_filename << endl;
      }
      auto engine_plan = infer_object(trt_engine->serialize());
      std::ofstream engine_file(engine_filename.c_str());
      engine_file.write((char *)engine_plan->data(), engine_plan->size());
      engine_file.close();

      {  // Test trt file, convert masknetweight from mask_filename to .trt
        CreateEngine(0, engine_filename.c_str(), mask_filename);
        RunEngine(0, max_batch_size, inputStream, 0, nullptr, 0);
        DestoryEngine(0);
      }
    }
  }
  if (inputStream) delete[] inputStream;

  if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
    cout << "All done" << endl;
  }
  ShowCudaMemInfo();

  return 0;
}
