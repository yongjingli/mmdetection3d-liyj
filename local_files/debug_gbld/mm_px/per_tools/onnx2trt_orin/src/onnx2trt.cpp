/*
 * Copyright (c) 2018~2020, Xpeng Motor. All rights reserved.
 * bin/onnx2trt_rcnn -o model_fp16b2.trt -b 2 -d 16  model.onnx
 * bin/onnx2trt_rcnn -o model_fp16b2.trt -b 2 -d 16  -i input.bin -p
 */

#include "NvOnnxParser.h"
#include "onnx_utils.hpp"
#include "common.hpp"
#include <onnx/optimizer/optimize.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <unistd.h> // For ::getopt
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <ctime>
#include <fcntl.h> // For ::open
#include <limits>
#include <list>

#include "ResizeBilinear.hpp"
#include "onnxtrt.h"
#include <NvInferRuntimeCommon.h>

#include <cuda_runtime_api.h>
#include "cuda_profiler_api.h"
#include <chrono>
#include "NvInfer.h"
#include <thread>

void printRegistedPlugins() {
  int plugin_nums = 0;
  IPluginCreator *const *plugins = getPluginRegistry()->getPluginCreatorList(&plugin_nums);
  printf("Registed %d plugins\n", plugin_nums);
  for (int i = 0; i < plugin_nums; i++) {
    printf("Registed plugin: %s\n", plugins[i]->getPluginName());
  }
}

// inner defines from onnx2trt.lib
extern int g_inputSize, g_outputSize;
extern bool debug_builder;
extern bool ifRunProfile;
size_t ReadBinFile(std::string filename, char *&databuffer);
int RunProfile(int ch, int b, char *i, int inputType, char *o, int outType);

using namespace nvinfer1;

// Logger for GIE info/warning/errors
class TRT_Logger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity _verbosity;
  std::ostream *_ostream;

 public:
  TRT_Logger(Severity verbosity = Severity::kWARNING, std::ostream &ostream = std::cout)
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char *msg) noexcept override {
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
                                                         : severity == Severity::kVERBOSE ? "VER" : "UNKNOWN");
      (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
    }
  }
};

inline void ShowCudaMemInfo() {
  size_t freeMem;
  size_t totalMem;
  static size_t lastFree;
  cudaMemGetInfo(&freeMem, &totalMem);
  DPRINTF(1, "CUDA MemInfo total= %zuBytes, free= %zuBytes, Delta= %zuBytes\n", totalMem, freeMem,
          (lastFree == 0) ? 0 : ((int64_t)lastFree - freeMem));
  lastFree = freeMem;
}

inline void ShowGPUInfo() {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  DPRINTF(1, "GPU %s @ %0.3f GHz, Compute Capability %d.%d\n", devProp.name, devProp.clockRate * 1e-6f, devProp.major,
          devProp.minor);
}

void print_version() {
  cout << "Parser built against:" << endl;
  cout << "  ONNX IR version:  " << onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << endl;
  cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << endl;
}

// code for modify output
class Node{
public:
  Node() {}
  Node(std::string name, std::string op_type)
    : name_(name)
    , op_type_(op_type)
    {}
  ~Node() = default;

  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }

  std::string op_type() { return op_type_; }
  void op_type(std::string op_type) { op_type_ = op_type; }

  Node* dependency(int i) { return dependencies_[i]; }
  std::vector<Node*> dependencies() { return dependencies_; }
  void add_dependency(Node* dependency) { dependencies_.push_back(dependency); }
  int dependencies_size() {return dependencies_.size(); }
  bool erase_dependency(Node* dependency) {
    std::vector<Node*>::iterator it;
    for (it = dependencies_.begin(); it != dependencies_.end();) {
      if (*it == dependency) {
        it = dependencies_.erase(it);
        return true;
      } else {
        it++;
      }
    }

    return false;
  }

  Node* dependent_node(int i) { return dependent_nodes_[i]; }
  std::vector<Node*> dependent_nodes() { return dependent_nodes_; }
  void add_dependent_node(Node* dependent_node) { dependent_nodes_.push_back(dependent_node); }
  int dependent_nodes_size() {return dependent_nodes_.size(); }
  bool erase_dependent_node(Node* dependent_node) {
    std::vector<Node*>::iterator it;
    for (it = dependent_nodes_.begin(); it != dependent_nodes_.end();) {
      if (*it == dependent_node) {
        it = dependent_nodes_.erase(it);
        return true;
      } else {
        it++;
      }
    }

    return false;
  }

private:
  std::string name_;
  std::string op_type_;
  std::vector<Node*> dependencies_;
  std::vector<Node*> dependent_nodes_;
};

class Edge{
public:
  Edge() {}
  Edge(std::string name)
    : name_(name)
    {}
  ~Edge() = default;

  bool is_valid() {
    if (from_ == nullptr || to_ == nullptr) {
      return false;
    }
    return true;
  }

  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }

  Node* from() { return from_; }
  void set_from(Node* from) { from_ = from; }

  Node* to() { return to_; }
  void set_to(Node* to) { to_ = to; }

private:
  std::string name_;
  Node* from_ = nullptr;
  Node* to_ = nullptr;
};

bool DestroyOneNode(Node* node, std::vector<Node *> *dependencies, std::list<Node> *nodes, std::list<Edge> *edges) {
  std::string node_name = node->name();
  int dependent_nodes_num = node->dependent_nodes_size();

  if (dependent_nodes_num != 0) {
    DPRINTF(2, "node \033[35m%s\033[0m has %d dependenct nodes, cannot destroy\n", node->name().data(), dependent_nodes_num);
    return false;
  }

  int dependencies_num = node->dependencies_size();
  dependencies->resize(dependencies_num);
  if (dependencies_num > 0) {
    memcpy(dependencies->data(), node->dependencies().data(), sizeof(Node *) * dependencies_num);
  }
  DPRINTF(4, "node \033[35m%s\033[0m has %d dependencies\n", node->name().data(), dependencies_num);

  // delete dependencies
  for (std::vector<Node*>::iterator it = dependencies->begin(); it != dependencies->end(); it++) {
    DPRINTF(4, "%s\n", (*it)->name().data());
    (*it)->erase_dependent_node(node);
  }

  // delete edge
  for (std::list<Edge>::iterator it = edges->begin(); it != edges->end();) {
    if ((*it).to()->name() == node->name()) {
      it = edges->erase(it);
    } else {
      it++;
    }
  }
  // delete node
  for (std::list<Node>::iterator it = nodes->begin(); it != nodes->end();) {
    if (&(*it) == node) {
      it = nodes->erase(it);
      break;
    } else {
      it++;
    }
  }

  DPRINTF(2, "destroy node \033[35m%s\033[0m successfully\n", node_name.data());

  return true;
}

// global params
struct Params {
  std::string engine, onnxModelFile, calibrationCache{"CalibrationTable"};
  int device{0}, batchSize{1}, workspaceSize{16}, DLACore{-1};
  bool fp16{false}, int8{false}, safe{false};
  float pct{99};
  Dims inputsize;
  DataType input_dtype;
} gParams;

// print the layers in network
void printLayers(nvinfer1::INetworkDefinition *network, nvinfer1::IBuilder *trt_builder, nvinfer1::IBuilderConfig *builder_config) {
  int layerNum = network->getNbLayers();
  int deviceType = 0;
  double totFlops = 0;
  double totWsize = 0;
  deviceType = (int)builder_config->getDefaultDeviceType();

  for (int i = 0; i < layerNum; i++) {
    auto layer = network->getLayer(i);
    deviceType = (int)builder_config->getDeviceType(layer);

    printf("Layer%d: %s DeviceType=%d\n", i, layer->getName(), deviceType);
    if (layer->getNbInputs() < 1) continue;

    auto type = layer->getType();
    auto inDim = layer->getInput(0)->getDimensions();
    for (int k = 0; k < layer->getNbOutputs(); k++) {
      auto outDim = layer->getOutput(k)->getDimensions();
      long long flops = 0;  // FLOPS
      long long wsize = 0;  // weight
      switch (type) {
        case LayerType::kCONVOLUTION: {
          auto kernelDim = ((IConvolutionLayer *)layer)->getKernelSize();
          int group = ((IConvolutionLayer *)layer)->getNbGroups();
          flops = (inDim.d[1] * kernelDim.d[0] * kernelDim.d[1] * 2 + 1) * volume(outDim) / group;
          wsize = inDim.d[1] * volume(kernelDim) * outDim.d[1] * sizeof(float) / group;
          printf("  FLOPS: %lld %dx(%d,%dx%d,%dx%dx%d) %lld B\n", flops, inDim.d[0], inDim.d[1], kernelDim.d[0], kernelDim.d[1],
                 outDim.d[1], outDim.d[2], outDim.d[3], wsize);
        } break;
        case LayerType::kFULLY_CONNECTED: {
          flops = volume(inDim) * (outDim.d[1] * 3);
          wsize = volume(inDim) * outDim.d[1] * sizeof(float);
          printf("  FLOPS: %lld %dx(%dx%dx%d,%dx%d) %lld B\n", flops, inDim.d[0], inDim.d[1], inDim.d[2], inDim.d[3], outDim.d[1],
                 outDim.d[2], wsize);
        } break;
        case LayerType::kMATRIX_MULTIPLY: {
          auto in1Dim = layer->getInput(1)->getDimensions();
          flops = volume(inDim) * (outDim.d[2] * 2);
          wsize = volume(in1Dim) * sizeof(float);
          printf("  FLOPS: %lld %dx(%dx%dx%d,%dx%d) %lld B\n", flops, inDim.d[0], inDim.d[1], inDim.d[2], inDim.d[3], outDim.d[1],
                 outDim.d[2], wsize);
        } break;
        case LayerType::kPOOLING: {
          auto kernelDim = ((IPoolingLayer *)layer)->getWindowSize();
          flops = (kernelDim.d[0] * kernelDim.d[1]) * volume(outDim);
          printf("  FLOPS: %lld %dx(%dx%d,%dx%dx%d) \n", flops, outDim.d[0], kernelDim.d[0], kernelDim.d[1], outDim.d[1], outDim.d[2],
                 outDim.d[2]);
        } break;
        case LayerType::kSCALE: { // batch-norm / scale
          flops = 3 * volume(outDim);
          printf("  FLOPS: %lld %dx(%d,%dx%dx%d) \n", flops, inDim.d[0], inDim.d[1], outDim.d[1], outDim.d[2], outDim.d[3]);
        } break;
        default:
          flops = volume(outDim);  // Sample estimate for
                                   // Relu,ElementWise,Shuffle,Plugin...
      }
      totFlops += flops;
      totWsize += wsize;
    }
  }
  printf("Tot %d Layers, %.3f GFlops, %.3f MB, DefaultDeviceType=%d\n", layerNum, totFlops / 1e9, totWsize / 1e6,
         deviceType);
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
    for (int i = 5; i < all - 5; i++) totTime += times[i];  // drop the fist & last 5 datas.

    printf(
        "TestAll %d range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< "
        "%.3fms\n",
        all, times[0], times[all - 1], totTime / (all - 10), times[all / 2], percentage, pctTime);
    return pctTime;
  }
  return std::numeric_limits<float>::infinity();
}

// return the size of image ( C*H*W ) , <0 error
int getPreprocesParams(int &modelType, int &im_w, int &im_h, int *idxRGB, float *Scales, float *Means) {
  if (gParams.inputsize.nbDims >= 3) {
    im_w = gParams.inputsize.d[2];
    im_h = gParams.inputsize.d[1];
  }

  // check for Width & Height
  char *val = getenv("TRT_IW");
  if (NULL != val) {
    im_w = atoi(val);
    printf("getenv TRT_IW=%d\n", im_w);
  }
  val = getenv("TRT_IH");
  if (NULL != val) {
    im_h = atoi(val);
    printf("getenv TRT_IH=%d\n", im_h);
  }

  val = getenv("TRT_BGR");
  if (NULL != val) {
    int is_BGR = atoi(val);
    printf("getenv TRT_BGR=%d\n", is_BGR);
    if (is_BGR) {
      idxRGB[0] = 2;
      idxRGB[2] = 0;
    }
  }

  if (1 == modelType) {
    Means[0] = 102.9801f;  // R
    Means[1] = 115.9465f;  // G
    Means[2] = 122.7717f;  // B
  }
  if (2 <= modelType) {
    Scales[0] = 255.f;  // R
    Scales[1] = 255.f;  // G
    Scales[2] = 255.f;  // B
  }

  if (7 == modelType) {          // nvidia-retinanet
    Scales[0] = 255.f * 0.229f;  // R
    Scales[1] = 255.f * 0.224f;  // G
    Scales[2] = 255.f * 0.225f;  // B
    Means[0] = 0.485f / 0.229f;  // R
    Means[1] = 0.456f / 0.224f;  // G
    Means[2] = 0.406f / 0.225f;  // B
  }

  return 0;
}

// ppm/pgm, jpeg, png
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// support multipy png/ppm images for batch>1, filenames:  batch0.png,batch1.jpg,batch2.ppm,...
int readPPMFile(const std::string filenames, char *&databuffer, int modelType, int batchsize = 1) {
  if (0 == filenames.compare(filenames.size() - 4, 4, ".bin")) {
    size_t size = ReadBinFile(filenames, databuffer);
    printf("ReadBinFile to inputStream size=%zu Bytes\n", size);
    return size;
  }

  // Get RGB and model intput
  int idxRGB[3] = {0, 1, 2};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  int model_w = 0;
  int model_h = 0;
  getPreprocesParams(modelType, model_w, model_h, idxRGB, Scales, Means);
  int qt_imgsize = model_w * model_h;
  DPRINTF(1, "Get Model input w=%d, h=%d batch=%d\n", model_w, model_h, batchsize);

  int size = -1;
  int iw, ih;
  int nb = 0;  // batch num (index)
  float *floatbuffers = nullptr;
  unsigned char *rgbbuffer = nullptr;

  std::stringstream ss;
  ss.str(filenames);
  std::string filename;
  for (; nb < batchsize && std::getline(ss, filename, ','); nb++) {
    int ch = 3;
    rgbbuffer = stbi_load(filename.c_str(), &iw, &ih, &ch, 0);
    // bool fourChInput = ch == 4 ? true : false;
    int real_ch = ch;
    ch = ch == 4 ? 3 : ch;
    if (nullptr == rgbbuffer) {
      break;
    }

    int imgsize = iw * ih;
    int real_size = imgsize * real_ch;
    size = imgsize * ch;
    DPRINTF(1, "batch[%d]:read %s by stbLib, size = %d (%dx%dx%d)\n", nb, filename.c_str(), real_size, real_ch, ih, iw);

    if (nullptr == floatbuffers) {
      floatbuffers = new float[size * batchsize];
    }
    float *floatbuffer = floatbuffers + nb * size;

    if(ch == 1) {
      for (int i = 0; i < imgsize; ++i) {
        for(int j = 0; j < ch; ++j) {
          reinterpret_cast<float*>(floatbuffer)[i + j * imgsize] = 1.f * (reinterpret_cast<char*>(rgbbuffer)[i * real_ch + idxRGB[j]]);
        }
      }
        
      float *pDst = new float[size];
      int src_idx=0;
      for( int c = 0; c < ch; c++){
        for( int y = 0; y < ih; y++) {
          for( int x = 0; x< iw; x++) {
            int dst_c = ((x & 1) * 2 + (y & 1)) * ch + c;
            int dst_y = y / 2;
            int dst_x = x / 2;
            int dst_idx = dst_c * (iw / 2) * (ih / 2) + dst_y * (iw / 2) + dst_x;    

            pDst[dst_idx] = floatbuffer[src_idx ++] / Scales[0];
          }
        }
      }

      memcpy( floatbuffer, pDst, size * sizeof(float));
      delete[] pDst;
    } else if (ch == 3) {
      for (int i = 0; i < imgsize; i++) {
        int o_i[3] = {i, i + imgsize, i + imgsize * 2};
        if (iw == model_w * 2) {
          int y = i / iw;
          int x = i % iw;

          int dst_c = ((x & 1) * 2 + (y & 1)) * ch;
          int dst_y = y / 2;
          int dst_x = x / 2;
          int dst_idx = dst_c * qt_imgsize + dst_y * model_w + dst_x;
          o_i[0] = dst_idx;
          o_i[1] = dst_idx + qt_imgsize;
          o_i[2] = dst_idx + 2 * qt_imgsize;
        }

        floatbuffer[o_i[0]] = (rgbbuffer[i * real_ch + idxRGB[0]] - Means[0]) / Scales[0];  // R
        floatbuffer[o_i[1]] = (rgbbuffer[i * real_ch + idxRGB[1]] - Means[1]) / Scales[1];  // G
        floatbuffer[o_i[2]] = (rgbbuffer[i * real_ch + idxRGB[2]] - Means[2]) / Scales[2];  // B
      }
      delete[] rgbbuffer;
    } else{
      DPRINTF(1, "Error: Invalid channel size detected as reading image.\n");
    }
  }

  // copy batch0 to rest batch.
  for (; nb < batchsize && size > 0; nb++) {
    memcpy(floatbuffers + nb * size, floatbuffers, size * sizeof(float));
  }

  databuffer = (char *)floatbuffers;

  return size * batchsize * sizeof(float);
}

// Read images from list for calibrator
class CalibratorFromList {
 public:
  CalibratorFromList(std::string listFile, std::string cacheFile, int mType = 4, int nBatch = 1)
      : mTotalSamples(0), mCurrentSample(0), mCacheFile(cacheFile) {
    modelType = mType;
    batchSize = nBatch;
    mTotalSamples = 1;  // default
    int imgsize = batchSize * volume(gParams.inputsize) / gParams.inputsize.d[0];
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
      DPRINTF(1, "Read %d ImageList from %s\n", mTotalSamples, listFile.c_str());
    } else
      DPRINTF(1, "Set const Data for Int8Calibrator\n");

    unsigned int eltSize = getElementSize(gParams.input_dtype);
    void *cudaBuffer;
    CHECK_CUDA(cudaMalloc(&cudaBuffer, imgsize * eltSize));
    CHECK_CUDA(cudaMemset(cudaBuffer, 0x40, imgsize * eltSize));  // 3.0f
    mInputDeviceBuffers.insert(std::make_pair(gInputs[1], cudaBuffer));
    gInputSize[1] = imgsize;

    // alloc the cuda memory for im_info ( 3 floats )
    CHECK_CUDA(cudaMalloc(&cudaBuffer, gInputSize[0] * sizeof(float)));
    mInputDeviceBuffers.insert(std::make_pair(gInputs[0], cudaBuffer));
  }

  virtual ~CalibratorFromList() {
    for (auto &elem : mInputDeviceBuffers) CHECK_CUDA(cudaFree(elem.second));
  }

  int getBatchSize() const { return batchSize; }

  bool getBatch(void *bindings[], const char *names[], int nbBindings) {
    if (mCurrentSample >= mTotalSamples) return false;

    if (0 < nbBindings) {
      int inputIndex = 0;
      if (2 == nbBindings) {
        DPRINTF(1, "getBatch[%d] nbBindings=%d %s %s\n", mCurrentSample, nbBindings, names[0], names[1]);
        bindings[0] = mInputDeviceBuffers[names[0]];

        inputIndex = 1;
      } else
        DPRINTF(1, "getBatch%d[%d] nbBindings=%d %s modelType=%d\n", batchSize, mCurrentSample, nbBindings, names[0],
                modelType);

      bindings[inputIndex] = mInputDeviceBuffers[gInputs[1]];
      int blocksize = gInputSize[1] * getElementSize(gParams.input_dtype);
      if (mImageList.size() > 0) {
        const char *input_filename = mImageList[mCurrentSample].c_str();
        if (strncmp("ConvLSTM:", input_filename, 9) == 0) {
          if (strstr(input_filename, "Video")) {
            DPRINTF(1, "ConvLSTM set Video Flag\n");
            SetConvLSTMState(0, 1);  // default Stream
          } else {
            DPRINTF(1, "ConvLSTM set Image Flag\n");
            SetConvLSTMState(0, 0);
          }

          ++mCurrentSample;
          if (mCurrentSample >= mTotalSamples) {
            return false;
          }
          input_filename = mImageList[mCurrentSample].c_str();
        }

        char *databuffer = nullptr;
        int ret = readPPMFile(input_filename, databuffer, modelType, batchSize);
        DPRINTF(1, "Read size %d vs Need size %d\n", ret, blocksize);
        if (ret >= gInputSize[1]) {
          CHECK_CUDA(cudaMemcpy(bindings[inputIndex], databuffer, blocksize, cudaMemcpyHostToDevice));
        } else {
          DPRINTF(1, "Error! readPPMFile %s size=%d, Request=%d. Skiped.\n", input_filename, ret, gInputSize[1]);
        }

        if (databuffer) {
          delete[] databuffer;
        }
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
      DPRINTF(1, "readCalibrationCache %s\n", mCacheFile.c_str());
      std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
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
  std::vector<int> gInputSize{3, 1 * 3 * 256 * 640};
};

class Int8Calibrator : public IInt8EntropyCalibrator {
 public:
  Int8Calibrator(std::string listFile, std::string cacheFile, int mType = 4, int nBatch = 1) {
    printf("Create %s\n", __func__);
    calibrator = new CalibratorFromList(listFile, cacheFile, mType, nBatch);
  }

  ~Int8Calibrator() { delete calibrator; }

  int getBatchSize() const noexcept override { return calibrator->getBatchSize(); }

  bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
    return calibrator->getBatch(bindings, names, nbBindings);
  }

  const void *readCalibrationCache(size_t &length) noexcept override { return calibrator->readCalibrationCache(length); }

  virtual void writeCalibrationCache(const void *cacheData, size_t dataSize) noexcept override {
    return calibrator->writeCalibrationCache(cacheData, dataSize);
  }

 private:
  CalibratorFromList *calibrator;
};

class DLACalibrator : public IInt8EntropyCalibrator2 {
 public:
  DLACalibrator(std::string listFile, std::string cacheFile, int mType = 4, int nBatch = 1) {
    printf("Create %s\n", __func__);
    calibrator = new CalibratorFromList(listFile, cacheFile, mType, nBatch);
  }

  ~DLACalibrator() { delete calibrator; }

  int getBatchSize() const noexcept override { return calibrator->getBatchSize(); }

  bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
    return calibrator->getBatch(bindings, names, nbBindings);
  }

  const void *readCalibrationCache(size_t &length) noexcept override { return calibrator->readCalibrationCache(length); }

  virtual void writeCalibrationCache(const void *cacheData, size_t dataSize) noexcept override {
    return calibrator->writeCalibrationCache(cacheData, dataSize);
  }

 private:
  CalibratorFromList *calibrator;
};

class MaxCalibrator : public IInt8MinMaxCalibrator {
 public:
  MaxCalibrator(std::string listFile, std::string cacheFile, int mType = 4, int nBatch = 1) {
    printf("Create %s\n", __func__);
    calibrator = new CalibratorFromList(listFile, cacheFile, mType, nBatch);
  }

  ~MaxCalibrator() { delete calibrator; }

  int getBatchSize() const noexcept override { return calibrator->getBatchSize(); }

  bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
    return calibrator->getBatch(bindings, names, nbBindings);
  }

  const void *readCalibrationCache(size_t &length) noexcept override { return calibrator->readCalibrationCache(length); }

  virtual void writeCalibrationCache(const void *cacheData, size_t dataSize) noexcept override {
    return calibrator->writeCalibrationCache(cacheData, dataSize);
  }

 private:
  CalibratorFromList *calibrator;
};

inline void setAllTensorScales(INetworkDefinition* network, float int8Scales = 2.0f){
  // Ensure that all layer inputs have a scale.
  for (int i = 0; i < network->getNbLayers(); i++){
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbInputs(); j++){
      ITensor* input{layer->getInput(j)};
      // Optional inputs are nullptr here and are from RNN layers.
      if (input != nullptr && !input->dynamicRangeIsSet()){
        input->setDynamicRange(-int8Scales, int8Scales);
      }
    }
    
    for (int j = 0; j < layer->getNbOutputs(); j++){
      ITensor* output{layer->getOutput(j)};
      // Optional outputs are nullptr here and are from RNN layers.
      if (output != nullptr && !output->dynamicRangeIsSet()){
        output->setDynamicRange(-int8Scales, int8Scales);
      }
    }        
  }
}

extern "C" void markuseroutput(const char *poutput);
extern "C" void markuserinput(const char *pinput);
extern "C" void markuserbatchsize(const size_t batch_size);

void print_usage() {
  cout << "ONNX to TensorRT model parser" << endl;
  cout << "Usage: onnx2trt onnx_model.pb" << "\n"
       << "                [-o engine_file.trt]  (output or test TensorRT engine)\n"
       << "                [-O outputlayers]  (name of output layer)(multiple)\n"
       << "                [-e engine_file.trt]  (test TensorRT engines(multiple), can be specified multiple times)\n"
       << "                [-i batch0.png,batch1.png]  (input data(multiple))\n"
       << "                [-t onnx_model.pbtxt] (output ONNX text file without weights)\n"
       << "                [-T onnx_model.pbtxt] (output ONNX text file with weights)\n"
       << "                [-m modelType|masknet.trt ](modelType=1,2,4)\n"
       << "                [-A onnx_model_out.pb] (output ONNX model)\n"
       << "                [-b max_batch_size (default 1)]\n"
       << "                [-B batch_size for int8 Calibration of combined model (default 1)]\n"
       << "                [-w max_workspace_size_bytes (default 16 MiB)]\n"
       << "                [-d model_data_type_bit_depth] (32 => float32, 16 => float16, 8 => int8)\n"
       << "                [-D DLA_layers_Number](support float16/int8)\n"
       << "                [-I Int8 layer index](1,8,16,32 => 1~8 & 16~32)\n"
       << "                [-S FP32 layer index](1,8,16,32 => 1~8 & 16~32)\n"
       << "                [-F frequency delay time/ms] set frequency delay time\n"
       << "                [-C imageList] (Cablition Int8 from ppm Image list)\n"
       << "                [-c Cablition Cache] (Cablition cache file)\n"
       << "                [-G modified_layers.txt] (modify output with -O option, then output modified layers)\n"
       << "                [-P passes] (optimize onnx model. Argument is a semicolon-separated list of passes)\n"
       << "                [-r referencefile] (check outdata with reference file, need -g)\n"
       << "                [-M] (Use IInt8MinMaxCalibrator)\n"
       << "                [-s] (safe mode)\n"
       << "                [-a] (list available optimization passes and exit)\n"
       << "                [-l] (list layers and their shapes)\n"
       << "                [-g] (debug mode, print outdata as floats)\n"
       << "                [-f] (optimize onnx model in fixed mode)\n"
       << "                [-v] (increase verbosity)\n"
       << "                [-q] (decrease verbosity)\n"
       << "                [-p] (enable profiler when run execution)\n"
       << "                [-V] (show version information)\n"
       << "                [-h] (show help)\n"
       << "Environment:\n"
       << "                TRT_DEBUGLEVEL: Control the Debug output, -1~4, default:1\n"
       << "                TRT_IC, TRT_IW, TRT_IH, TRT_ID: Input CHW and Type(8/16/32)\n"
       << "                TRT_GEMV: Enable Int8GEMV and set TopK of weight, 1~2, default:-1\n"
       << "                TRT_SAVEPATH: path for saving input/output data of inference\n"
       << "                TRT_UPSAMPLE: plugin version of upsamples, default:2. Resize:0, plugin_v2:2\n"
       << "                TRT_SLICE: version of slices, default:1. onnx-tensorrt:0, our reformed:1\n";
}

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  ShowGPUInfo();
  ShowCudaMemInfo();
  printRegistedPlugins();

  std::string engine_filename;
  std::vector<std::string> engine_filenames;
  std::string input_filename;
  std::vector<std::string> input_filenames;
  std::string model_filename;
  std::string text_filename;
  std::string optimization_passes_string;
  std::vector<std::string> optimizationPassNames;
  std::string full_text_filename;
  size_t max_batch_size = 1;
  size_t calib_batch = 1;               // batch size for INT8 calibrator
  size_t max_workspace_size = 1 << 27;  // default: 128MiB, Parking > 128MB
  int model_dtype_nbits = 32;
  int model_DLA_Layer_Num = 0;        // the layes Num of DLA, [0,TotLayers)
  std::vector<int> model_INT8_Layer;  // the begin-end index of layes for int8
  std::vector<int> model_FP32_Layer;  // the begin-end index of layes for fp32
  std::vector<int> model_FP16_Layer;  // the begin-end index of layes for fp16
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  bool optimize_model = false;
  bool optimize_model_fixed = false;
  bool print_optimization_passes_info = false;
  bool print_layer_info = false;
  int modelType = 4;                          // 1: maskrcnn, 4: ldd&ds 2: retinanet
  const char *mask_filename = "Task=LLDMOD";  //"fastrcnn_nomaskfile";
  std::string Calibrator_imgList;
  float fps_delay = 30;           // delay ms to get fixed FPS
  std::string outRefer_filename;  // reference output file, checked with -g
  nvinfer1::CalibrationAlgoType calibrator_type = nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION; // 1: Entropy, 2: Entropy2, 3: MaxMin
  std::string modified_layers_file;
  std::vector<std::string> modified_output;

  {
    int low, high;
    cudaDeviceGetStreamPriorityRange(&low, &high);
    DPRINTF(2, "StreamPriorityRange:[%d,%d]\n", low, high);
    DPRINTF(2, "sizeof cudaStream_t = %ld\n", sizeof(cudaStream_t));
    DPRINTF(2, "sizeof void* = %ld\n", sizeof(void *));
    DPRINTF(2, "sizeof EngineBuffer = %ld\n", sizeof(EngineBuffer));
  }

  int arg = 0;
  while((arg = ::getopt(argc, argv, "o:e:i:b:B:w:t:T:D:m:A:d:I:S:F:C:c:G:O:r:P:plagfvqVhMs")) != -1) {
    if (strchr("o:e:i:b:B:w:t:T:D:m:A:d:I:S:F:C:c:G:O:r:P:", arg)) {
      if (!optarg) {
        cerr << "ERROR: -" << arg << " flag requires argument" << endl;
        return -1;
      }
    }
    
    switch (arg){
      case 'o':
        engine_filename = optarg;
        engine_filenames.push_back(optarg);
        break;
      case 'e':
        engine_filenames.push_back(optarg);
        break;
      case 'i':
        input_filename = optarg;
        input_filenames.push_back(optarg);
        break;
      case 'b':
        max_batch_size = atoll(optarg);
        markuserbatchsize(max_batch_size);
        break;
      case 'B':
        calib_batch = atoll(optarg);
        break;
      case 'w':
        max_workspace_size = atoll(optarg);
        break;
      case 't':
        text_filename = optarg;
        break;
      case 'T':
        full_text_filename = optarg;
        break;
      case 'D':
        calibrator_type = nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2; // EntroyCalibrator2
        model_DLA_Layer_Num = atoi(optarg);
        break;
      case 'm':
        if (strlen(optarg) == 1) {
          modelType = atoi(optarg);
        } else {
          modelType = 1;
          mask_filename = optarg;
        }
        break;
      case 'A':
        model_filename  = optarg;
        break;
      case 'd':
        model_dtype_nbits = atoi(optarg);
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
      case 'C':  //
        Calibrator_imgList = optarg;
        break;
      case 'c':  //
        gParams.calibrationCache = optarg;
        break;
      case 'G':
        modified_layers_file = optarg;
        break;
      case 'O':
        markuseroutput(optarg);
        modified_output.push_back(optarg);
        break;
      case 'r':
        outRefer_filename = optarg;
        break;
      case 'P':
        optimize_model = true;
        optimization_passes_string = optarg;
        break;
      case 'p':
        ifRunProfile = true;
        break;
      case 'l':
        print_layer_info = true;
        break;
      case 'a':
        print_optimization_passes_info = true;
        break;
      case 'g':
        debug_builder = true;
        break;
      case 'f':
        optimize_model_fixed = true;
        optimize_model = true;
        break;
      case 'v':
        ++verbosity;
        ++TRT_DEBUGLEVEL;
        break;
      case 'q':
        --verbosity;
        --TRT_DEBUGLEVEL;
        break;
      case 'V':
        print_version();
        return 0;
      case 'h':
        print_usage();
        return 0;
      case 'M':
        calibrator_type = nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION; // MinMax Calibrator
        break;
      case 's':
        gParams.safe = true;
        break;
    }
  }

  if (optimize_model || print_optimization_passes_info) {
    optimizationPassNames = ::ONNX_NAMESPACE::optimization::GetAvailablePasses();
  }

  if (print_optimization_passes_info) {
    cout << "Available optimization passes are:" << endl;
    for( auto it = optimizationPassNames.begin(); it != optimizationPassNames.end(); it++ )
    {
      cout << " " << it->c_str() << endl;
    }
    return 0;
  }

  TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto trt_builder = common::infer_object(nvinfer1::createInferBuilder(trt_logger));
  auto trt_network = common::infer_object(trt_builder->createNetworkV2(explicitBatch));

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
    cerr << "ERROR: Invalid model data type bit depth: " << model_dtype_nbits << endl;
    return -2;
  }

  char *inputStream{nullptr};

  int num_args = argc - optind;
  int ch_num = engine_filenames.size();
  int outType = (modelType == 1 || modelType == 2) ? 1 : 0;
  if(num_args != 1) {
    if (ch_num == 0){
      print_usage();
      return -1;
    } else if (!engine_filename.empty()) {
      // load directly from serialized engine file if onnx model not specified
      cudaProfilerStop();

      int ret = CreateEngine(0, engine_filename.c_str(), mask_filename);
      if( ret < 0 ) {
        DPRINTF(1, "onnx2trt Cannot CreateEngine %s\n", engine_filename.c_str());
        return -1;
      }

      {
        int bufferNum = 0;
        EngineBuffer *bufferInfo;
        GetBufferOfEngine(0, &bufferInfo, &bufferNum, NULL);
        if (bufferNum > 1) {
          gParams.inputsize.nbDims = bufferInfo[0].nDims;
          memcpy(gParams.inputsize.d, bufferInfo[0].d, sizeof(int) * 4);
        }
      }

      if (!input_filename.empty()) {
        readPPMFile(input_filename, inputStream, modelType, max_batch_size);
      }

      RunEngine(0, max_batch_size, inputStream, 0, nullptr, outType);
      cudaProfilerStart();
      {
        int output_size = g_outputSize;
        char *output;
        if (!outRefer_filename.empty()) {
          output_size = ReadBinFile(outRefer_filename, output);
          outType |= 0x100;  // check reference output
        } else {
          output = new char[output_size];
        }

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
        chID[ch] = CreateEngine(-1, engine_filenames[ch].c_str(), mask_filename);
        float ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();

        printf("[%d]CreateEngine CHID.%d time = %fms\n", ch, chID[ch], ms);
        output[ch] = new char[g_outputSize];
        RunEngine(chID[ch], max_batch_size, inputStream, 0, output[ch], outType);
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

          float ms =
              std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
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
        std::thread t[ch_num];
        for (int ch = 0; ch < ch_num; ch++) {
          t[ch] = std::thread(testRunEngine, chID[ch], max_batch_size, inputStream, 0, output[ch], outType);
        }
        for (int ch = 0; ch < ch_num; ch++) {
          t[ch].join();
        }
#endif
        float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start)
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

    if (!std::ifstream(onnx_filename.c_str())) {
      cerr << "Input file not found: " << onnx_filename << endl;
      return -3;
    }

    ::ONNX_NAMESPACE::ModelProto _the_onnx_model;
    ::ONNX_NAMESPACE::ModelProto& onnx_model = _the_onnx_model;
    bool is_binary = common::ParseFromFile_WAR(&onnx_model, onnx_filename.c_str());
    if (!is_binary && !common::ParseFromTextFile(&onnx_model, onnx_filename.c_str())) {
      cerr << "Failed to parse ONNX model" << endl;
      return -3;
    }

    if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
      int64_t opset_version = (onnx_model.opset_import().size() ?
                              onnx_model.opset_import(0).version() : 0);
      cout << "----------------------------------------------------------------" << endl;
      cout << "Input filename:   " << onnx_filename << endl;
      cout << "ONNX IR version:  " << common::onnx_ir_version_string(onnx_model.ir_version()) << endl;
      cout << "Opset version:    " << opset_version << endl;
      cout << "Producer name:    " << onnx_model.producer_name() << endl;
      cout << "Producer version: " << onnx_model.producer_version() << endl;
      cout << "Domain:           " << onnx_model.domain() << endl;
      cout << "Model version:    " << onnx_model.model_version() << endl;
      cout << "Doc string:       " << onnx_model.doc_string() << endl;
      cout << "----------------------------------------------------------------" << endl;
    }

    if (onnx_model.ir_version() > ::ONNX_NAMESPACE::IR_VERSION) {
      cerr << "WARNING: ONNX model has a newer ir_version ("
          << common::onnx_ir_version_string(onnx_model.ir_version())
          << ") than this parser was built against ("
          << common::onnx_ir_version_string(::ONNX_NAMESPACE::IR_VERSION) << ")." << endl;
    }

    if (!modified_layers_file.empty()) {
      // decode graph
      std::list<Node> nodes;
      std::list<Edge> edges;
      std::set<std::string> data_dict;
      int input_num  = onnx_model.graph().input_size();
      int init_num   = onnx_model.graph().initializer_size();
      int output_num = onnx_model.graph().output_size();
      int node_num   = onnx_model.graph().node_size();

      // load data_dict from input and initializer
      for (int i = 0; i < input_num; i++) {
        std::string input_name = onnx_model.graph().input(i).name();
        if (input_name == "image") {
          continue;
        }
        data_dict.insert(input_name);
      }

      for (int i = 0; i < init_num; i++) {
        std::string init_name = onnx_model.graph().initializer(i).name();
        data_dict.insert(init_name);
      }

      // load node name
      Node input("image", "input");
      nodes.push_back(input);
      for (int i = 0; i < node_num; i++) {
        std::string node_name = onnx_model.graph().node(i).name();
        std::string op_type = onnx_model.graph().node(i).op_type();
        Node node(node_name, op_type);
        nodes.push_back(node);
      }
      for (int i = 0; i < output_num; i++) {
        std::string output_name = onnx_model.graph().output(i).name();
        Node output(output_name, "output");
        nodes.push_back(output);
      }

      // connect edge
      // add dependency
      std::list<Node>::iterator it_node = nodes.begin();
      it_node++;
      for (int i = 0; i < node_num; i++) {
        int dependency_num = onnx_model.graph().node(i).input_size();
        for (int j = 0; j < dependency_num; j++) {
          std::string dependency_name = onnx_model.graph().node(i).input(j);
          if (data_dict.count(dependency_name) == 0) {
            Edge edge(dependency_name);
            edge.set_to(&(*it_node));
            edges.push_back(edge);
          }
        }
        it_node++;
      }
      for (int i = 0; i < output_num; i++) {
        std::string dependency_name = onnx_model.graph().output(i).name();
        Edge edge(dependency_name);
        edge.set_to(&(*it_node));
        edges.push_back(edge);
        it_node++;
      }

      // add dependent node
      it_node = nodes.begin();
      {
        std::string dependent_node_name = "image";
        for (std::list<Edge>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++) {
          if (dependent_node_name == (*it_edge).name()) {
            (*it_edge).set_from(&(*it_node));
            (*it_node).add_dependent_node((*it_edge).to());
            (*it_edge).to()->add_dependency(&(*it_node));
          }
        }
      }
      it_node++;
      for (int i = 0; i < node_num; i++) {
        int dependent_node_num = onnx_model.graph().node(i).output_size();
        for (int j = 0; j < dependent_node_num; j++) {
          std::string dependent_node_name = onnx_model.graph().node(i).output(j);
          for (std::list<Edge>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++) {
            if (dependent_node_name == (*it_edge).name()) {
              (*it_edge).set_from(&(*it_node));
              (*it_node).add_dependent_node((*it_edge).to());
              (*it_edge).to()->add_dependency(&(*it_node));
            }
          }
        }
        it_node++;
      }

      // record modified output
      std::queue<Node *> nodeQueue;
      std::set<Node *> existNode;
      it_node = nodes.begin();
      for (int i = 0; i < node_num + 1; i++) {
        it_node++;
      }
      for (int i = 0; i < output_num; i++) {
        bool flag = false;
        for (size_t j = 0; j < modified_output.size(); j++) {
          if ((*it_node).name() == modified_output[j]) {
            flag = true;
          }
        }
        if (modified_output.size() == 0)
          flag = true;
        
        if (flag) {
          std::cout << "remain output[" << i << "] " << (*it_node).name() << std::endl;
        } else {
          std::cout << "delete output[" << i << "] " << (*it_node).name() << std::endl;
          nodeQueue.push(&(*it_node));
          existNode.insert(&(*it_node));
        }
        it_node++;
      }

      // delete output
      std::vector<Node *> dependencies;
      while (!nodeQueue.empty()) {
        Node* node = nodeQueue.front();
        DestroyOneNode(node, &dependencies, &nodes, &edges);
        for (size_t i = 0; i < dependencies.size(); i++) {
          if (!existNode.count(dependencies[i])) {
            nodeQueue.push(dependencies[i]);
            existNode.insert(dependencies[i]);
          }
        }
        
        existNode.erase(node);
        nodeQueue.pop();
      }
      std::cout << "remain " << nodes.size() << " layers of onnx model" << std::endl;

      // output modified layers
      std::fstream out_file(modified_layers_file, std::ios::out);
      if (out_file.is_open()) {
        it_node = nodes.begin();
        for (it_node = nodes.begin(); it_node != nodes.end(); it_node++) {
          if ((*it_node).op_type() == "Reshape") {
            for (std::list<Edge>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++) {
              if ((*it_edge).to()->name() == (*it_node).name()) {
                out_file << (*it_edge).name() << " copy" << std::endl;
              }
            }
          }
          if ((*it_node).op_type() != "Concat") {
            out_file << (*it_node).name() << std::endl;
          } else {
            for (std::list<Edge>::iterator it_edge = edges.begin(); it_edge != edges.end(); it_edge++) {
              if ((*it_edge).to()->name() == (*it_node).name()) {
                out_file << (*it_edge).name() << " copy" << std::endl;
              }
            }
          }
        }
      }
      out_file.close();

      return -6;
    }

    if (!model_filename.empty()) {
      if( optimize_model ) {
        std::vector<std::string> passes;

        std::string curPass;
        std::stringstream passStream(optimization_passes_string);
        while( std::getline(passStream, curPass, ';') ) {
          if( std::find(optimizationPassNames.begin(), optimizationPassNames.end(), curPass) != optimizationPassNames.end() ) {
            passes.push_back(curPass);
          }
        }

        if (!passes.empty()) {
          cout << "Optimizing '" << model_filename << "'" << endl;
          ::ONNX_NAMESPACE::ModelProto _the_onnx_model_optimized = optimize_model_fixed
                                                                ? ::ONNX_NAMESPACE::optimization::OptimizeFixed(onnx_model, passes)
                                                                : ::ONNX_NAMESPACE::optimization::Optimize(onnx_model, passes);
          onnx_model = _the_onnx_model_optimized;
        }
      }

      if (!common::MessageToFile(&onnx_model, model_filename.c_str())) {
        cerr << "ERROR: Problem writing ONNX model" << endl;
      }
    }

    if (!text_filename.empty()) {
      if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        cout << "Writing ONNX model (without weights) as text to " << text_filename << endl;
      }
      std::ofstream onnx_text_file(text_filename.c_str());
      std::string onnx_text = pretty_print_onnx_to_string(onnx_model);
      onnx_text_file.write(onnx_text.c_str(), onnx_text.size());
    }
    if (!full_text_filename.empty()) {
      if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        cout << "Writing ONNX model (with weights) as text to " << full_text_filename << endl;
      }
      std::string full_onnx_text;
      google::protobuf::TextFormat::PrintToString(onnx_model, &full_onnx_text);
      std::ofstream full_onnx_text_file(full_text_filename.c_str());
      full_onnx_text_file.write(full_onnx_text.c_str(), full_onnx_text.size());
    }

    auto trt_parser = common::infer_object(nvonnxparser::createParser(*trt_network, trt_logger));

    if (verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING) {
      cout << "Parsing model" << endl;
    }

    std::ifstream onnx_file(onnx_filename.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);
    if (!onnx_file.read(onnx_buf.data(), onnx_buf.size())) {
      cerr << "ERROR: Failed to read from file " << onnx_filename << endl;
      return -4;
    }
    if (!trt_parser->parse(onnx_buf.data(), onnx_buf.size())) {
      int nerror = trt_parser->getNbErrors();
      for( int i=0; i<nerror; ++i ) {
        nvonnxparser::IParserError const* error = trt_parser->getError(i);
        if( error->node() != -1 ) {
          ::ONNX_NAMESPACE::NodeProto const& node =
            onnx_model.graph().node(error->node());
          cerr << "While parsing node number " << error->node()
              << " [" << node.op_type();
          if( node.output().size() ) {
            cerr << " -> \"" << node.output(0) << "\"";
          }
          cerr << "]:" << endl;
          if( verbosity >= (int)nvinfer1::ILogger::Severity::kINFO ) {
            cerr << "--- Begin node ---" << endl;
            cerr << node << endl;
            cerr << "--- End node ---" << endl;
          }
        }
        cerr << "ERROR: "
            << error->file() << ":" << error->line()
            << " In function " << error->func() << ":\n"
            << "[" << static_cast<int>(error->code()) << "] " << error->desc()
            << endl;
      }
      return -5;
    }

    auto builder_config = common::infer_object(trt_builder->createBuilderConfig());
    if (print_layer_info)  // print the layers in network
    {
      printLayers(trt_network.get(), trt_builder.get(), builder_config.get());
    }

    if (!engine_filename.empty()) {
      gParams.inputsize = trt_network.get()->getInput(0)->getDimensions();
      gParams.input_dtype = trt_network.get()->getInput(0)->getType();
      if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        DPRINTF(1, "Platform support FP16:%d, INT8:%d\n", fp16, int8);
        DPRINTF(1, "Building TensorRT engine, \n");
        DPRINTF(1, "\tInputSize: [%d,%d,%d,%d]\n", gParams.inputsize.d[0], gParams.inputsize.d[1], gParams.inputsize.d[2], gParams.inputsize.d[3]);
        DPRINTF(1, "\tInput data type size: %d\n", getElementSize(gParams.input_dtype));
        DPRINTF(1, "\tDataType(-d): %d bits\n", model_dtype_nbits);
        DPRINTF(1, "\tMax batch(-b): %zu\n", max_batch_size);
        DPRINTF(1, "\tMax workspace(-w): %zu B\n", max_workspace_size);
      }

      int outNum = trt_network.get()->getNbOutputs();
      for( int i=0; i< outNum; i++){
        DPRINTF(1, "\tOutput[%d](-O): %s\n", i, trt_network.get()->getOutput(i)->getName());
      }

      // auto trt_profile = trt_builder->createOptimizationProfile();
      // auto name = trt_network.get()->getInput(0)->getName();
      // trt_profile->setDimensions(name, OptProfileSelector::kMIN, Dims4{1, gParams.inputsize.d[1], gParams.inputsize.d[2], gParams.inputsize.d[3]});
      // trt_profile->setDimensions(name, OptProfileSelector::kOPT, Dims4{max_batch_size, gParams.inputsize.d[1], gParams.inputsize.d[2], gParams.inputsize.d[3]});
      // trt_profile->setDimensions(name, OptProfileSelector::kMAX, Dims4{max_batch_size, gParams.inputsize.d[1], gParams.inputsize.d[2], gParams.inputsize.d[3]});
      // builder_config->addOptimizationProfile(trt_profile);

      std::unique_ptr<IInt8Calibrator> calibrator;
      builder_config->setMaxWorkspaceSize(max_workspace_size);
      if( fp16 && model_dtype == nvinfer1::DataType::kHALF) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
      } else if( model_dtype == nvinfer1::DataType::kINT8 ) {
        if (nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2 == calibrator_type) {
          if (0 > model_DLA_Layer_Num) { // PDK6.0.2 safe DLA has error with DLACalibrator.
            setAllTensorScales(trt_network.get());
          } else {
            calibrator.reset(new DLACalibrator(Calibrator_imgList, gParams.calibrationCache, modelType, calib_batch));
          }
        } else if (nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION == calibrator_type) {
          calibrator.reset(new MaxCalibrator(Calibrator_imgList, gParams.calibrationCache, modelType, calib_batch));
        } else {
          calibrator.reset(new Int8Calibrator(Calibrator_imgList, gParams.calibrationCache, modelType, calib_batch));
        }
        if (0 < model_DLA_Layer_Num) {
          builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        builder_config->setInt8Calibrator(calibrator.get());
      }

      // builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      builder_config->setDLACore(0);
      if (0 < model_DLA_Layer_Num) {
        builder_config->setDefaultDeviceType(DeviceType::kDLA);
        builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
      }
      if (0 > model_DLA_Layer_Num) {  // Use safe model, all layer run on DLA
        trt_builder->setMaxBatchSize(max_batch_size);
        builder_config->setDefaultDeviceType(DeviceType::kDLA);
        builder_config->setEngineCapability(EngineCapability::kDLA_STANDALONE);
        int tfmt = (model_dtype == nvinfer1::DataType::kINT8) ? 0x20 : 0x10;
        int inNum = trt_network->getNbInputs();
        for (int i = 0; i < inNum; i++) {
          auto input = trt_network->getInput(i);
          input->setType(model_dtype);
          input->setAllowedFormats(tfmt);
        }
        int outNum = trt_network->getNbOutputs();
        for (int i = 0; i < outNum; i++) {
          auto output = trt_network->getOutput(i);
          output->setType(model_dtype);
          output->setAllowedFormats(tfmt);
        }
      } else {
        builder_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);  // Use GPU
      }

      int int8SliceNum = model_INT8_Layer.size();
      int fp32SliceNum = model_FP32_Layer.size();
      if (int8SliceNum > 0) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
      }
      if (int8SliceNum > 0 || fp32SliceNum > 0) {
        builder_config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
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
            if (i >= model_INT8_Layer[j] && i <= model_INT8_Layer[j + 1]) isINT8Layer = true;
          }
          for (int j = 0; j < fp32SliceNum; j += 2) {
            if (i >= model_FP32_Layer[j] && i <= model_FP32_Layer[j + 1]) isFP32Layer = true;
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
        } else {
          // Don't set the precision on non-computation layers as they don't support int8.
          if (layer->getType() != LayerType::kCONSTANT && layer->getType() != LayerType::kCONCATENATION
              && layer->getType() != LayerType::kSHAPE) {
            if (model_dtype == nvinfer1::DataType::kINT8) {
              // set computation precision of the layer
              layer->setPrecision(nvinfer1::DataType::kINT8);
              DPRINTF(2, "set kINT8 at layer[%d]%s\n", i, pName);
            } else if (model_dtype != nvinfer1::DataType::kFLOAT) {
              layer->setPrecision(nvinfer1::DataType::kHALF);
              DPRINTF(2, "set kHALF at layer[%d]%s\n", i, pName);
            }
          }
        }

        // set the layes befor model_DLA_Layer_Num to run DLA
        if (model_DLA_Layer_Num != 0 && model_DLA_Layer_Num != -1) {
          if (i <= abs(model_DLA_Layer_Num)) {
            builder_config->setDeviceType(layer, DeviceType::kDLA);
            DPRINTF(1, "setDeviceType kDLA at layer[%d]%s\n", i, pName);
          } else {
            builder_config->setDeviceType(layer, DeviceType::kGPU);
            DPRINTF(1, "setDeviceType kGPU at layer[%d]%s\n", i, pName);
          }
        }
      }

      // builder_config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
      // trt_builder->setMaxBatchSize(max_batch_size);
      auto trt_engine = common::infer_object(trt_builder->buildEngineWithConfig(*trt_network.get(), *builder_config.get()));
      calibrator.reset();

      auto engine_plan = common::infer_object(trt_engine->serialize());
      std::ofstream engine_file(engine_filename.c_str());
      if (!engine_file) {
        cerr << "Failed to open output file for writing: "
            << engine_filename << endl;
        return -6;
      }
      if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
        cout << "Writing TensorRT engine to " << engine_filename << endl;
      }
      engine_file.write((char*)engine_plan->data(), engine_plan->size());
      engine_file.close();

      if (model_DLA_Layer_Num >= 0) {  // Not support saft DLA, model_DLA_Layer_Num = -1
        // Test trt file, convert masknetweight from mask_filename to .trt
        CreateEngine(0, engine_filename.c_str(), mask_filename);
        RunEngine(0, max_batch_size, nullptr, 0, nullptr, 0);
        RunProfile(0, max_batch_size, nullptr, 0, nullptr, 0);
        DestoryEngine(0);
      }
    }
  }
  
  if (inputStream) delete[] inputStream;

  if( verbosity >= (int)nvinfer1::ILogger::Severity::kWARNING ) {
    cout << "All done" << endl;
  }

  ShowCudaMemInfo();

  return 0;
}
