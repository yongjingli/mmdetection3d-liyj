/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * libonnxtrt.so
 */
#include <fcntl.h>   // For ::open
#include <unistd.h>  // For ::getopt
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "ResizeBilinear.hpp"  // Resize for upsample in Feng's Model
//#include "conv_relu.h"         //For mask-net of maskrcnn
#include "onnxtrt.h"
#include "VoxelGenerator.h"


//extern int post_nms_topN;  // support to change at converting and adjust at inference.
extern int prob_num;       // add by dyg

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
                     : severity == Severity::kWARNING ? "WARNING"
                                                      : severity == Severity::kINFO ? "   INFO" : "UNKNOWN");
      (*_ostream) << "[" << buf << " " << sevstr << "] " << msg << std::endl;
    }
  }
};

TrtCudaGraph::~TrtCudaGraph() {
  if (mGraphExec) {
    cudaGraphExecDestroy(mGraphExec);
  }
  if (mGraph) {
    cudaGraphDestroy(mGraph);
  }
}

void TrtCudaGraph::beginCapture(cudaStream_t& stream) {
  DPRINTF(2, "begin capture:\n");

  cudaCheck(cudaGraphCreate(&mGraph, 0));
  cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void TrtCudaGraph::launch(cudaStream_t& stream) {
  cudaCheck(cudaGraphLaunch(mGraphExec, stream));
}

void TrtCudaGraph::endCapture(cudaStream_t& stream) {
  cudaCheck(cudaStreamEndCapture(stream, &mGraph));
  cudaCheck(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));

  updateNodesAndEdges();

  DPRINTF(2, "end capture with %zu nodes and %zu edges\n", numNodes, numEdges);

  if (TRT_DEBUGLEVEL >= 4) {
      printGraph();
  }
}

bool TrtCudaGraph::deleteNodes(std::vector<int> id) {
  DPRINTF(2, "begin delete %zu nodes:\n", id.size());

  for (size_t i = 0; i < id.size(); i++) {
    DPRINTF(2, "delete node %d\n", id[i]);
    if (!destroyOneNode(id[i])) {
      DPRINTF(2, "delete node %d\n failure", id[i]);
      return false;
    }
  }

  updateGraphExec();

  updateNodesAndEdges();

  DPRINTF(1, "end delete with %zu nodes and %zu edges\n", numNodes, numEdges);

  return true;
}

bool TrtCudaGraph::destroyOneNode(int i) {
  std::vector<cudaGraphNode_t> pDependencies;
  size_t pNumDependencies;
  cudaCheck(cudaGraphNodeGetDependencies(nodes[i], nullptr, &pNumDependencies));
  if (pNumDependencies == 0) {
    DPRINTF(2, "node %d : %p is root node\n", i, nodes[i]);
    pDependencies.resize(0);
  } else {
    pDependencies.resize(pNumDependencies);
    cudaCheck(cudaGraphNodeGetDependencies(nodes[i], pDependencies.data(), &pNumDependencies));
  }

  std::vector<cudaGraphNode_t> pDependentNodes;
  size_t pNumDependentNodes;
  cudaCheck(cudaGraphNodeGetDependentNodes(nodes[i], nullptr, &pNumDependentNodes));
  if (pNumDependentNodes == 0) {
    DPRINTF(2, "node %d : %p is end node\n", i, nodes[i]);
    pDependentNodes.resize(0);
  } else {
    pDependentNodes.resize(pNumDependentNodes);
    cudaCheck(cudaGraphNodeGetDependentNodes(nodes[i], pDependentNodes.data(), &pNumDependentNodes));
  }

  if (pNumDependencies > 1 || pNumDependentNodes > 1) {
    DPRINTF(2, "node %d : %p has %zu dependencies and %zu dependent nodes\n", i, nodes[i], pNumDependencies, pNumDependentNodes);
    return false;
  }

  if (pNumDependencies == 1 && pNumDependentNodes == 1) {
    cudaCheck(cudaGraphDestroyNode(nodes[i]));

    cudaCheck(cudaGraphAddDependencies(mGraph, &pDependencies[0], &pDependentNodes[0], pNumDependencies));
  } else {
    cudaCheck(cudaGraphDestroyNode(nodes[i]));
  }

  DPRINTF(2, "destroy node %d successfully\n", i);
  return true;
}

void TrtCudaGraph::updateGraphExec() {
  cudaCheck(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
}

void TrtCudaGraph::updateNodesAndEdges() {
  cudaCheck(cudaGraphGetNodes(mGraph, nullptr, &numNodes));
  if (numNodes > 0) {
    nodes.resize(numNodes);
    cudaCheck(cudaGraphGetNodes(mGraph, nodes.data(), &numNodes));
  }

  cudaCheck(cudaGraphGetEdges(mGraph, nullptr, nullptr, &numEdges));
  if (numNodes > 0) {
    from.resize(numEdges);
    to.resize(numEdges);
    cudaCheck(cudaGraphGetEdges(mGraph, from.data(), to.data(), &numEdges));
  }
}

void TrtCudaGraph::printGraph() {
  std::cout << "printGraph" << std::endl;
  updateNodesAndEdges();

  printf("printGraph numNodes=%zu numEdges=%zu\n", numNodes, numEdges);
  for (size_t i = 0; i < numNodes; i++) {
    cudaGraphNodeType sType;
    struct CUfunc_st *pFunc;
    std::string printHead;

    cudaCheck(cudaGraphNodeGetType(nodes[i], &sType));

    size_t pNumDependencies = 0;
    cudaCheck(cudaGraphNodeGetDependencies(nodes[i], nullptr, &pNumDependencies));

    size_t pNumDependentNodes = 0;
    cudaCheck(cudaGraphNodeGetDependentNodes(nodes[i], nullptr, &pNumDependentNodes));

    printf("node[%zu] address=%p type=%d numDependencies=%zu numDependentNode=%zu\n", i, nodes[i], sType, pNumDependencies, pNumDependentNodes);
    cudaKernelNodeParams kernelNodeParams;
    cudaMemcpy3DParms memcpy3DParms;
    cudaMemsetParams memsetParams;
    switch(sType) {
      case cudaGraphNodeTypeKernel:
        error = cudaGraphKernelNodeGetParams(nodes[i], &kernelNodeParams);
        if (error != cudaSuccess) {
            std::cout << "cudaGraphKernelNodeGetParams Error " << error << std::endl;
        }
        
        pFunc=(struct CUfunc_st *)kernelNodeParams.func;
        char head[20];
        sprintf(head, "Node%zu", i);
        printHead = head;
        printFuncInfo(pFunc, printHead, 0, 0, 1);
        // printf( " kernelNodeParams: %p, %d, %d, %d, %p\n", pFunc, kernelNodeParams.gridDim.x, kernelNodeParams.blockDim.x,
                                                    // kernelNodeParams.sharedMemBytes, kernelNodeParams.kernelParams);
      break;
      case cudaGraphNodeTypeMemcpy:
        error = cudaGraphMemcpyNodeGetParams(nodes[i], &memcpy3DParms);
        if (error != cudaSuccess) {
            std::cout << "cudaGraphMemcpyNodeGetParams Error " << error << std::endl;
        }

        std::cout << "memcpy kind is " << memcpy3DParms.kind << std::endl;
      break;
      case cudaGraphNodeTypeMemset:
        error = cudaGraphMemsetNodeGetParams(nodes[i], &memsetParams);
        if (error != cudaSuccess) {
            std::cout << "cudaGraphMemsetNodeGetParams Error " << error << std::endl;
        }
      break;
      default:
        printf("node[%zu] type=%d not support yet!\n", i, (int)sType);
      break;
    }
  }

  for (size_t i = 0; i < numEdges; i++) {
    std::cout << "edge " << i << " from node " << from[i] << " to " << to[i] << std::endl;
  }
}

void Dump(const void * mem, unsigned int n) {
  const char * p = reinterpret_cast<const char *>(mem);
  int j = 0;
  int k = 0;
  long offset = 0;
  std::stringstream sHex, sAscii;
  for (unsigned int i = 0; i < n; i++) {
    int q = int(p[i] & 0xff);
    sHex << std::setfill('0') << std::setw(2) << std::hex << q << " ";
    if (j == 7) {
      sHex << " "; 
    }
    if (q > 31 && q < 127) { 
      sAscii << (char)q ;
    } else {
      sAscii << ".";
    }
    j++;
    if (j == 16) {
      unsigned long index = int(offset + (k * 16));
      unsigned long addrs = index + (unsigned long)mem;
      std::cout << std::setfill('0') << std::setw(8) << std::hex << addrs <<"|"<< std::setfill('0') << std::setw(8) << std::hex << index << ": " << sHex.str() << "|" << sAscii.str() << "|" << std::endl;
      sHex.str("");
      sAscii.str("");
      j = 0;
      k++;
    }
  }
  std::cout << std::endl;
}

bool isBadPtr(const char* p) {
  int fh = open(p, 0, 0);
  int e = errno;

  if (-1 == fh && e == EFAULT) {
    return true;
  } else if (fh != -1) {
    close(fh);
  }
  return false;
}

void printFuncInfo(struct CUfunc_st *pFunc, std::string printHead, bool if_dump, bool if_ptr, bool if_value) {
  struct kernel *pKernel=pFunc->kernel;

  std::cout << "---------------" << printHead << "---------------" << std::endl;
  
  if (if_dump) {
    struct myf *pmyf=(struct myf *)pFunc;
    Dump((void *)pmyf, sizeof(struct myf));
  }

  if (if_ptr) {
    std::cout << "Partially decoded CUfunc_st struct" << std::endl;
    std::cout << "func pointer address:\t\t" << pFunc << std::endl;
    std::cout << "v0 pointer address:\t\t" << &pFunc->v0 << std::endl;
    std::cout << "v1 pointer address:\t\t" << &pFunc->v1 << std::endl;
    std::cout << "name pointer address:\t\t" << &pFunc->name << std::endl;
    std::cout << "v2 pointer address:\t\t" << &pFunc->v2 << std::endl;
    std::cout << "v3 pointer address:\t\t" << &pFunc->v3 << std::endl;
    std::cout << "v4 pointer address:\t\t" << &pFunc->v4 << std::endl;
    std::cout << "v5 pointer address:\t\t" << &pFunc->v5 << std::endl;
    std::cout << "kernel pointer address:\t\t" << &pFunc->kernel << std::endl;

    std::cout << "Partially decoded kernel struct" << std::endl;
    std::cout << "kernel.v0 pointer address:\t" << &pKernel->v0 << std::endl;
    std::cout << "kernel.v1 pointer address:\t" << &pKernel->v1 << std::endl;
    std::cout << "kernel.v2 pointer address:\t" << &pKernel->v2 << std::endl;
    std::cout << "kernel.v3 pointer address:\t" << &pKernel->v3 << std::endl;
    std::cout << "kernel.v4 pointer address:\t" << &pKernel->v4 << std::endl;
    std::cout << "kernel.v5 pointer address:\t" << &pKernel->v5 << std::endl;
    std::cout << "kernel.v6 pointer address:\t" << &pKernel->v6 << std::endl;
    std::cout << "kernel.v7 pointer address:\t" << &pKernel->v7 << std::endl;
    std::cout << "kernel.v8 pointer address:\t" << &pKernel->v8 << std::endl;
    std::cout << "kernel.size pointer address:\t" << &pKernel->size << std::endl;
    std::cout << "kernel.module pointer address:\t" << &pKernel->module << std::endl;
    std::cout << "kernel.v9 pointer address:\t" << &pKernel->v9 << std::endl;
  }

  if (if_value) {
    std::cout << "v0:\t\t" << pFunc->v0 << std::endl;
    std::cout << "v1:\t\t" << pFunc->v1 << std::endl;
    if (!isBadPtr(pFunc->name)) {
      std::cout << "name:\t\t" << pFunc->name << std::endl;
    } else {
      std::cout << "func->name is a bad pointer" << std::endl;
    }
    std::cout << "v2:\t\t" << pFunc->v2 << std::endl;
    std::cout << "v3:\t\t" << pFunc->v3 << std::endl;
    std::cout << "v4:\t\t" << pFunc->v4 << std::endl;
    std::cout << "v5:\t\t" << pFunc->v5 << std::endl;
    
    if (!isBadPtr((char*)(pFunc->kernel))) {
      std::cout << "kernel.v0:\t" << pKernel->v0 << std::endl;
      std::cout << "kernel.v1:\t" << pKernel->v1 << std::endl;
      std::cout << "kernel.v2:\t" << pKernel->v2 << std::endl;
      std::cout << "kernel.v3:\t" << pKernel->v3 << std::endl;
      std::cout << "kernel.v4:\t" << pKernel->v4 << std::endl;
      std::cout << "kernel.v5:\t" << pKernel->v5 << std::endl;
      std::cout << "kernel.v6:\t" << pKernel->v6 << std::endl;
      std::cout << "kernel.v7:\t" << pKernel->v7 << std::endl;
      std::cout << "kernel.v8:\t" << pKernel->v8 << std::endl;
      std::cout << "kernel.v9:\t" << pKernel->v9 << std::endl;
      std::cout << "kernel.size: " << pKernel->size << " size of the module" << std::endl;
      std::cout << "kernel.module: " << pKernel->module << " pointer to the module (the ELF binary)" << std::endl;
    } else {
      std::cout << "func->kernel is a bad pointer" << std::endl;
    }
  }

  std::cout << "----------------------------------------------------------\n";
}

bool debug_builder = false;
bool ifRunProfile = false;
int TIMING_ITERATIONS = 1;
struct Profiler : public nvinfer1::IProfiler {
  typedef std::pair<std::string, float> Record;
  std::vector<Record> mProfile;

  virtual void reportLayerTime(const char *layerName, float ms)
#ifdef ORIN_UBUNTU // Need on Orin
  noexcept
#endif
  {
    auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record &r) { return r.first == layerName; });
    if (record == mProfile.end())
      mProfile.push_back(std::make_pair(layerName, ms));
    else
      record->second += ms;
  }

  void printLayerTimes() {
    float totalTime = 0;
    for (size_t i = 0; i < mProfile.size(); i++) {
      printf("%-150.150s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
      totalTime += mProfile[i].second;
    }
    printf("Time over %ld layers: %4.3f\n", mProfile.size(), totalTime / TIMING_ITERATIONS);
  }

} gProfiler;

// Refer from tensorrt4.0 trtexec.cpp
#include <cuda_runtime_api.h>
#include <chrono>
#include "NvInfer.h"

using namespace nvinfer1;

inline void ShowCudaMemInfo() {
  size_t freeMem;
  size_t totalMem;
  static size_t lastFree;
  cudaMemGetInfo(&freeMem, &totalMem);
  std::cout << "CUDA MemInfo total = " << totalMem << "Bytes, free = " << freeMem
            << "Bytes, Delta = " << (int64_t)lastFree - (int64_t)freeMem << "Bytes" << std::endl;
  lastFree = freeMem;
}

size_t ReadBinFile(std::string filename, char *&databuffer) {
  size_t size = 0;
  FILE *fp = fopen(filename.c_str(), "rb");
  if (fp) {
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    databuffer = new char[size];
    size = fread(databuffer, 1, size, fp);
    fclose(fp);
  }
  return size;
}

template <typename DType>
void printOutput(int64_t memSize, void *buffer) {
  int eltCount = memSize / sizeof(DType);
  DType *outputs = new DType[eltCount];
  CHECK_CUDA(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

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

struct GpuTimer {
  void *start;
  void *stop;

  GpuTimer() {
    cudaEvent_t *p_start = (cudaEvent_t *)&start;
    cudaEvent_t *p_stop = (cudaEvent_t *)&stop;
    cudaEventCreate(p_start);
    cudaEventCreate(p_stop);
  }
  ~GpuTimer() { 
    cudaEventDestroy((cudaEvent_t)start);
    cudaEventDestroy((cudaEvent_t)stop);
  }

  void Start() { 
    cudaEventRecord((cudaEvent_t)start, 0); 
  }
  void Stop(){ 
    cudaEventRecord((cudaEvent_t)stop, 0);
  }
  float Elapsed() { 
    float elapsed;
    cudaEventSynchronize((cudaEvent_t)stop);
    cudaEventElapsedTime(&elapsed, (cudaEvent_t)start, (cudaEvent_t)stop);
    return elapsed;
  }
};

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
  int fsd_out_num;  // Found FSD out num

  int frameID = 0;
  std::string savePath = "";  // save input data to path
  std::string loadFile = "";  // load input data from file
  std::string constResultFile = "";  // const result data from file

  int buffer_tot_size = 0;
  int buffer_block_size[ONNXTRT_MAX_BUFFERTYPE];
  std::vector<int64_t> calculateBindingBufferSizes(int batchSize);

  int priority = 0;
  bool useCudaGraph = false;
  bool hasImplicitBatchDimension = false;

  // bool cpInCudaGraph = false; // copy output in CudaGraph
  TrtCudaGraph *mGraphs[ONNXTRT_MAX_BUFFERNUM];  // Use CudaGraph to reduce CPU usage
  std::string cudaGraphNodeFile = "";
  std::vector<std::vector<int>> nodeLists; // ID of deleted nodes for different batch size

  //lidar params
  VoxelGenerator* _voxelGen = nullptr;
  bool _isVgInitialzed = false;

  
 public:
  //gpu timer
  GpuTimer* _gpuTimer = nullptr;
  bool  _isGtInitialized = false;
  float _timeAccumulator = 0.0f;
  int   _iterationTimes = 0;
  bool  _warmedUp = false;

  IWorkspace() {
    memset(bufferInfo, 0, sizeof(bufferInfo));
    memset(mGraphs, 0, sizeof(mGraphs));
    nodeLists.clear();
  }
  ~IWorkspace() { release(); };

  void setStatus(int s) {
    cudaEngineCheck(ch, status);
    status = s;
  }
  bool isInited() { return (nullptr != engine && status > 0); }
  int init(int setch, char *trtModelStream, size_t size, const char *config);

  void doInference(char *inputData = NULL, char *outData = NULL, int batchSize = 1, int runTest = 1,
                   int inputType = 0, int outType = 0);

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
    cudaMemcpyAsync(cpuBuf, gpuBuf, size, cudaMemcpyDeviceToHost, stream);
    char outfile[260];
    snprintf(outfile, 256, "%s/ch%d_%d.bin", savePath.c_str(), ch, frameID);
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file.write(cpuBuf, size);
    file.close();
    delete[] cpuBuf;

    DPRINTF(2, "Save input %s size=%ld\n", outfile, size);
    return size;
  }

  int saveOutBuf(void *cpuBuf, int64_t size) {
    if (cpuBuf == nullptr || size <= 0) return -1;

    char outfile[260];
    snprintf(outfile, 256, "%s/out_ch%d_%d.bin", savePath.c_str(), ch, frameID);
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file.write((const char *)cpuBuf, size);
    file.close();

    DPRINTF(2, "Save output %s size=%ld\n", outfile, size);
    return size;
  }

  int loadBuf(std::string &inFile, void *outBuf, int64_t size, int devType) {
    if (outBuf == nullptr || size < 0) return -1;

    char *cpuBuf = (char*)outBuf;
    if( devType ) { // If on GPU
      cpuBuf = new char[size]; 
    }

    std::ifstream file(inFile, std::ios::in | std::ios::binary);
    if (!file.good()) {
      DPRINTF(1, "load input %s Error\n", loadFile.c_str());
      return -1;
    }

    if( 0 == size ) { // Read all d
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
    }
    file.read(cpuBuf, size);
    int outsize = file.tellg();
    file.close();
    if( devType ) { // If on GPU
      cudaMemcpyAsync(outBuf, cpuBuf, size, cudaMemcpyHostToDevice, stream);
      delete[] cpuBuf;
    }

    DPRINTF(2, "load input %s size=%d\n", inFile.c_str(), outsize);
    return outsize;
  }

  bool readDeletedNodes(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::in);
    if (!file.is_open()) {
      return false;
    }

    nodeLists.resize(0);
    char lineBuf[2048] = {0};
    while (file.getline(lineBuf, sizeof(lineBuf))) {
      std::stringstream word(lineBuf);
      int id;
      std::vector<int> list;
      while (word >> id) {
        list.push_back(id);
      }
      nodeLists.push_back(list);
    }

    return true;
  }
};

const int MAX_BATCH = ONNXTRT_MAX_BATCHNUM;
const int MAX_CH = ONNXTRT_MAX_ENGINENUM;
static IWorkspace gworkspace[MAX_CH];         // first model for this engine.
static IWorkspace gworkspace_second[MAX_CH];  // second model with different batch of this engine.

int g_inputSize, g_outputSize;
std::vector<int64_t> IWorkspace::calculateBindingBufferSizes(int batchSize) {
  std::vector<int64_t> sizes;
  for (int ti = 0; ti < ONNXTRT_MAX_BUFFERTYPE; ti++) {
    buffer_block_size[ti] = 0;
  }

  EngineBuffer *secondBufferInfo = nullptr;
  int secondNum = 0;
  void *secondStream = nullptr;  
  if (gworkspace_second[ch].isInited()) {
    gworkspace_second[ch].getBufferInfo(&secondBufferInfo, &secondNum, &secondStream);
  }  


  for (int i = 0; i < nbBindings; ++i) {
    bool isInput = engine->bindingIsInput(i);
    const char *name = engine->getBindingName(i);
    Dims dims = engine->getBindingDimensions(i);
    DataType dtype = engine->getBindingDataType(i);

    if(!engine->hasImplicitBatchDimension()){
      dims.nbDims -= 1;  // Skip batch
      for (int k = 0; k < dims.nbDims; k++) {
        dims.d[k] = dims.d[k+1];
      }
      for (int k = dims.nbDims; k < ONNXTRT_MAX_BUFFERDIMS; k++) {
        dims.d[k] = 0;
      }
    }

    int64_t bufferSize = volume(dims) * getElementSize(dtype);
    sizes.push_back(bufferSize);

    bufferInfo[i].name = name;
    bufferInfo[i].nBufferType = (isInput) ? 0 : (strchr(name, '/') ? 2 : 1);
    bufferInfo[i].nDataType = (int)dtype;
    bufferInfo[i].nMaxBatch = batchSize;
    bufferInfo[i].nDims = dims.nbDims;
    memcpy(bufferInfo[i].d, dims.d, sizeof(int) * dims.nbDims);
    bufferDims[i] = dims;

    for (int j = 0; j < secondNum; j++){
      if (0 == strncmp(secondBufferInfo[j].name, name, ONNXTRT_MAX_BUFFERNUM)) {
        bufferInfo[i].nMaxBatch = std::max(secondBufferInfo[j].nMaxBatch, batchSize);
        break;
      }
    }
      
    const char *sBufferType[ONNXTRT_MAX_BUFFERTYPE] = {"In:", "Out:", "Var:"};
    DPRINTF(1, "Binding[%d] %s%s %dx[%d,%d,%d],%u\n", i, sBufferType[bufferInfo[i].nBufferType], name,
            bufferInfo[i].nMaxBatch, dims.d[0], dims.d[1], dims.d[2], getElementSize(dtype));

    buffer_tot_size += (bufferSize * batchSize);
    buffer_block_size[bufferInfo[i].nBufferType] += (bufferSize * batchSize);
  }
  g_inputSize = buffer_block_size[0];
  g_outputSize = buffer_block_size[1] + buffer_block_size[2];
  DPRINTF(1, "TRT_INSIZE=%dB \nTRT_OUTSIZE=%dB (Variable=%dB)\n", g_inputSize, g_outputSize, buffer_block_size[2]);

  char databuf[33];
  snprintf(databuf, sizeof(databuf), "%d", g_inputSize);
  setenv("TRT_INSIZE", databuf, 1);
  snprintf(databuf, sizeof(databuf), "%d", g_outputSize);
  setenv("TRT_OUTSIZE", databuf, 1);

  return sizes;
}

int IWorkspace::init(int setch, char *trtModelStream, size_t size, const char *config) {
  if (nullptr == trtModelStream) return -1;
  DPRINTF(2, "initializeing Iwork space\n");

  size_t size_trt = *(int *)(trtModelStream + 16) + 0x18;
  DPRINTF(1, "TRT engine written size %zu\n", size_trt);
  if (size < size_trt) {
    DPRINTF(1, "TRT engine written size %zu > TRT engine file size %zu\n", size_trt, size);
  }

  trt_logger = new TRT_Logger(nvinfer1::ILogger::Severity::kWARNING);
  DPRINTF(2, "Creating InferRuntime\n");
  infer = createInferRuntime(*trt_logger);
  DPRINTF(2, "InferRuntime Created\n");

  engine = infer->deserializeCudaEngine(trtModelStream, size_trt, nullptr);
  DPRINTF(2, "CUDA engine deserialized\n");

  if (NULL == engine) {
    return ONNXTRT_IO_ERR;
  }

  hasImplicitBatchDimension = engine->hasImplicitBatchDimension();
  DPRINTF(1, "EngineHasImplictBatchDimension = %d\n", hasImplicitBatchDimension);

  {
    char *measureTimeChar = getenv("TRT_MEASURETIME");
    int measureTime = 0;
    if(NULL != measureTimeChar)
      measureTime = atoi(measureTimeChar);
    
    
    if(measureTime == 1) {
      DPRINTF(2, "initializeing gpu timer\n");
      _gpuTimer = new GpuTimer;
      _isGtInitialized = true;
       DPRINTF(2, "gpu timer is initialized\n");
    }
  }

  DPRINTF(1, "Loading Engine, info = %s\n Engine config = %s\n", engine->getName(), config ? config : "NULL");
  ch = setch;

  {  // get config from env
    char *val = getenv("TRT_SAVEPATH");
    if (NULL != val) {
      savePath = val;
      DPRINTF(2, "TRT_SAVEPATH = %s\n", savePath.c_str());
    }
    val = getenv("TRT_LOADFILE");
    if (NULL != val) {
      loadFile = val;
      DPRINTF(2, "TRT_LOADFILE = %s\n", loadFile.c_str());
    }
  }

  if (nullptr != config) {  // get config from config string
    priority = (ch == 0) ? -1 : 0;
    if (strstr(config, "Priority=")) {
      if (strstr(config, "Priority=High")) {
        priority = -1;
      } else {
        priority = 0;
      }
      DPRINTF(1, "[CH%d] Priority = %d\n", ch, priority);
    }
    if (strstr(config, "CudaGraph=True")) {
      useCudaGraph = true;
      DPRINTF(1, "[CH%d] useCudaGraph = %s\n", ch, useCudaGraph ? "True" : "False");

      bool readEnd = 0;
      size_t nodePtr = size_trt;
      if (size > size_trt + 3) {
        char title[4];
        strncpy(title, (char *)(trtModelStream + nodePtr), 4);
        nodePtr += 4;
        if (title[0] == 'L' && title[1] == 'i' && title[2] == 's' && title[3] == 't') {
          readEnd = 1;
        } else {
          DPRINTF(1, "[CH%d] Info title %s is not 'List', will not load cuda graph list info at the end of model\n", ch, title);
        }
      }
      if (readEnd) {
        size_t list_total_len = *(int *)(trtModelStream + nodePtr);
        if (size < list_total_len + nodePtr) {
          DPRINTF(1, "List written size %zu > List file size %zu\n", list_total_len, size - size_trt);
        }
        nodePtr += 4;
        int batch_num = *(int *)(trtModelStream + nodePtr);
        nodePtr += 4;
        DPRINTF(2, "CUDA graph deleted node batch number = %d\n", batch_num);
        nodeLists.resize(batch_num);
        for (int i = 0; i < batch_num; i++) {
          int list_len = *(int *)(trtModelStream + nodePtr);
          nodePtr += 4;
          DPRINTF(2, "CUDA graph deleted node list length = %d\n", list_len);
          nodeLists[i].resize(list_len);
          for (int j = 0; j < list_len; j++) {
            int list_id = *(int *)(trtModelStream + nodePtr);
            nodePtr += 4;
            nodeLists[i][j] = list_id;
          }
        }
        if (size == nodePtr) {
          DPRINTF(1, "[CH%d] Load CUDA graph node lists at the end of engine successfully\n", ch);
        } else {
          DPRINTF(1, "[CH%d] Warning: engine size + node lists size != file size\n", ch);
        }
      } else {
        const char *val = strstr(config, "CudaGraphNodeFile");
        if (NULL != val) {
          int i = 18;
          while (val[i] != ',' && val[i] != '\0') {
            cudaGraphNodeFile += val[i];
            i++;
          }
          DPRINTF(1, "[CH%d] CudaGraphNodeFile = %s\n", ch, cudaGraphNodeFile.c_str());
          if (!readDeletedNodes(cudaGraphNodeFile)) {
            DPRINTF(1, "[CH%d] use default Cuda Graph\n", ch);
          } else {
            DPRINTF(1, "[CH%d] find CudaGraphNodeFile, successfully read deleted node list in %s\n", ch, cudaGraphNodeFile.c_str());
          }
        } else {
          DPRINTF(1, "[CH%d] use default Cuda Graph\n", ch);
        }
      }
    }
    if (strstr(config, ".bin")) {
      constResultFile = config;
    }
  }
  CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
  context = engine->createExecutionContext();

  {
    char databuf[33];
    nbBindings = engine->getNbBindings();
    assert(nbBindings <= ONNXTRT_MAX_BUFFERNUM);
    buffers.resize(nbBindings);
    bufferIndex.resize(nbBindings);
    if(engine->hasImplicitBatchDimension()){
      nMaxBatchsize = engine->getMaxBatchSize();
    } else {
      nMaxBatchsize = engine->getBindingDimensions(0).d[0];
    }
    nMaxBatchsize = std::min(nMaxBatchsize, MAX_BATCH);
    buffersSizes = calculateBindingBufferSizes(nMaxBatchsize);

    // Malloc a plan memory, reduce memcpy call number for I/O.
    CHECK_CUDA(cudaMalloc(&bufferInfo[0].p, buffer_tot_size));
    for (int i = 0; i < nbBindings; ++i) {
      bufferInfo[i].p = bufferInfo[0].p;
      bufferInfo[i].nBufferSize = (buffersSizes[i] * nMaxBatchsize);
      bufferIndex[i] = i;
      buffers[bufferIndex[i]] = bufferInfo[i].p;
    }

  _isVgInitialzed = false;
  if(NULL != strcasestr(bufferInfo[0].name, "lidar_pcd")) {
    if(0 == memcmp(bufferInfo[0].name, "vg:", 3)){
      std::vector<float> pcRangeUpper(3, 0.0f);
      std::vector<float> pcRangeLower(3, 0.0f);
      std::vector<float> resolution  (3, 0.0f);
      float scale;
      const char *param_str = bufferInfo[0].name + 3;
      // vg:/R_xx_xx_xx/X_xx_xx/Y_xx_xx/Z_xx_xx/S_xx
      sscanf(param_str, "/R_%f_%f_%f/X_%f_%f/Y_%f_%f/Z_%f_%f/S_%f", &resolution[0], 
            &resolution[1], &resolution[2], &pcRangeLower[0], &pcRangeUpper[0],
            &pcRangeLower[1], &pcRangeUpper[1], &pcRangeLower[2], &pcRangeUpper[2],
            &scale);

      _voxelGen = new VoxelGenerator(pcRangeUpper, pcRangeLower, resolution, 
                                     stream, bufferInfo[0].nDataType, scale);

      DPRINTF(2, "vg params: /R_%f_%f_%f/X_%f_%f/Y_%f_%f/Z_%f_%f/S_%f\n", resolution[0], 
            resolution  [1], resolution  [2], pcRangeLower[0], pcRangeUpper[0],
            pcRangeLower[1], pcRangeUpper[1], pcRangeLower[2], pcRangeUpper[2],
            scale);

      bufferInfo[0].name = "lidar_pcd";

    }else {
      _voxelGen = new VoxelGenerator(stream, bufferInfo[0].nDataType, 255.0f);
    }
    _voxelGen->initWorkSpace();
    _isVgInitialzed = true;
  }

    snprintf(databuf, sizeof(databuf), "%d", bufferDims[bufferIndex[0]].d[1]);
    setenv("TRT_IH", databuf, 0);
    snprintf(databuf, sizeof(databuf), "%d", bufferDims[bufferIndex[0]].d[2]);
    setenv("TRT_IW", databuf, 0);
  }

  DPRINTF(2, "Init nbBindings=%d\n", nbBindings);

  return 0;
}

inline bool isAlmostEqual(float x, float y, int ulp = 2) {
  return std::fabs(x - y) < std::numeric_limits<float>::epsilon() * std::fabs(x + y) * ulp ||
         std::fabs(x - y) < std::numeric_limits<float>::min();
}

inline void checkOutput(float *outData, float *outRef, int size, int ulp) {
  int count_notEq = 0;
  for (int i = 0; i < size; i++) {
    if (!isAlmostEqual(outData[i], outRef[i], ulp)) {
      DPRINTF(1, "%d Not Equal: %f vs %f\n", i, outData[i], outRef[i]);
      if (++count_notEq >= 16) break;
    };
  }
  DPRINTF(1, "checkOutput count_notEq = %d\n", count_notEq);
}


void writeResultBin(float *opData, int writeNum) {
    // const int imageSize = imageWidth * imageHeight;
  
    unsigned int valNum = static_cast<unsigned int>(writeNum);
    std::string outFileName ="./test.bin";
    std::ofstream out(outFileName, std::ios_base::binary);
    if(out.good()) {
        out.write((const char*)opData, valNum *  sizeof(float));
        out.close();
    }else {
        cout<<"Error: Cannot write to binary file. Return."<<endl;
    }
}


#include "data_convert.h"
// inputType: 0:cpu float32, 1:gpu float32, 2: gpu 3HW (CHW) float32 3: gpu 3HW (CHW) int8 4:lidar CPU pcd
// 5:cpu 3HW uint8 
// outType: 0:cpu roi & feat map, 1: cpu roi & mask
void IWorkspace::doInference(char *inputData, char *outData, int batchSize, int runTest, int inputType, int outType) {
  if (nMaxBatchsize != batchSize && gworkspace_second[ch].isInited()) {
    return gworkspace_second[ch].doInference(inputData, outData, batchSize, runTest, inputType, outType);
  }

  batchSize = std::min(nMaxBatchsize, batchSize);  // check batchsize

  // Use one GPU buffer for all input and output
  int cur_block_size[ONNXTRT_MAX_BUFFERTYPE] = {
      0,
  };
  char *buffer_block_ptr[ONNXTRT_MAX_BUFFERTYPE] = {
      (char *)bufferInfo[0].p,
  };
  for (int ti = 1; ti < ONNXTRT_MAX_BUFFERTYPE; ++ti) {
    cur_block_size[ti] = 0;
    buffer_block_ptr[ti] = buffer_block_ptr[ti - 1] + buffer_block_size[ti - 1];
  }
  {  // float offset of In/Out/Variable Data
    for (int ib = 0; ib < nbBindings; ++ib) {
      int type = bufferInfo[ib].nBufferType;
      bufferInfo[ib].p = (void *)(buffer_block_ptr[type] + cur_block_size[type]);
      buffers[bufferIndex[ib]] = bufferInfo[ib].p;
      cur_block_size[type] += buffersSizes[bufferIndex[ib]] * batchSize;
    }
    DPRINTF(2, "Use one buffer size=(%d, %d, %d)Byte\n", cur_block_size[0], cur_block_size[1], cur_block_size[2]);
  }
  int tot_out_size = cur_block_size[1];
  
  // Get the index of input and set the evn of input if undefined
  int inputIndex = bufferIndex[0];
  // Check if copy GPU buffer or use pointer directly( GPU or Uniform memory )
  if (inputType >= 0 && nullptr != inputData) {
    int num = buffersSizes[inputIndex] * batchSize;
    if(2 == inputType || 3 == inputType) {
        // buffers[inputIndex] = inputData;
        buffers[inputIndex] = bufferInfo[0].p;
        int inDataType = (2 == inputType)? 0 : 2; 
        TransFormat(inputData, batchSize, 3, inDataType, bufferInfo, stream);
        //Res2ChannelWrapperInt8(inputData, (char*)buffers[inputIndex], dataNum, batchSize, 3, width * 2, height * 2, stream);
    } else if(4 == inputType) { 
      if(!_isVgInitialzed) {
        printf("Error: The input format is pcd, but VG is not initialized. Retrun.\n");
        return;
      }
      DPRINTF(2, "start generating voxels\n");
      buffers[inputIndex] = bufferInfo[0].p;
      LidarInput *lidarInput = reinterpret_cast<LidarInput*>(inputData);
      void* pointCloud = lidarInput->pointCloud;
      // cout<<"lidar data num is "<<lidarInput->pointNum<<" points"<<endl; 
      _voxelGen -> copyData(reinterpret_cast<PointXYZI*>(pointCloud), lidarInput->pointNum);
      _voxelGen -> generateVoxels(buffers[inputIndex]); 

      DPRINTF(2, "voxel generation finish\n");

    } else if(5 == inputType) {
      DPRINTF(2,"The copy num as IO is %d bytes.", num);
      buffers[inputIndex] = bufferInfo[0].p;
      CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], inputData, num, cudaMemcpyHostToDevice, stream));
    } else if (0 == (inputType & 1)) {  // bit 0
      buffers[inputIndex] = bufferInfo[0].p;
      CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], inputData, num, cudaMemcpyHostToDevice, stream));
    } else  {
        buffers[inputIndex] = inputData;
    }
    
    SetConvLSTMState(stream, (inputType >> 16));  // bit 1

    frameID++;
    DPRINTF(2, "Feed %d Input[%d] size=%d type=%d\n", frameID, inputIndex, num, inputType);
    if (!loadFile.empty()) loadBuf(loadFile, buffers[inputIndex], num, 1);
    if (!savePath.empty()) saveGPUBuf(buffers[inputIndex], num);
  }

  if (0 == runTest && batchSize > 0) {
    if (inputType >= 0) {
      if (!useCudaGraph) {
        if (hasImplicitBatchDimension) {
          context->enqueue(batchSize, &buffers[0], stream, nullptr);
        } else {
          context->enqueueV2(&buffers[0], stream, nullptr);
        }
      } else {
        auto &pGraph = mGraphs[batchSize];
        if (nullptr == pGraph) {  // record from stream at first run
          pGraph = new TrtCudaGraph();
          if (hasImplicitBatchDimension) {
            context->enqueue(batchSize, &buffers[0], stream, nullptr);
          } else {
            context->enqueueV2(&buffers[0], stream, nullptr);
          }
          cudaStreamSynchronize(stream);

          pGraph->beginCapture(stream);
          if (hasImplicitBatchDimension) {
            context->enqueue(batchSize, &buffers[0], stream, nullptr);
          } else {
            context->enqueueV2(&buffers[0], stream, nullptr);
          }
          pGraph->endCapture(stream);
          if ((int)nodeLists.size() >= batchSize) {
            pGraph->deleteNodes(nodeLists[batchSize - 1]);
          } else {
            DPRINTF(2, "won't delete any nodes\n");
          }
        } else {  // launch CudaGraph, reduce CPU usage
          pGraph->launch(stream);
        }
      }
    }

    int offset = 0;  // offset of OutData while outbuf num > 1 orbatch > 1
    if (nullptr != outData) {
      DPRINTF(2, "Fetch Output size=%d type=%d\n", cur_block_size[1], outType);
      auto copy_type = cudaMemcpyDeviceToHost;
      if (outType >= 0) {  // sync here to avoid cudaMemcpyAsync blocking preprocessing
        cudaStreamSynchronize(stream);
      } else if (outType == -2) {  // output buffer is Managed Memory
        copy_type = cudaMemcpyDeviceToDevice;
      }
      CHECK_CUDA(cudaMemcpyAsync(outData, buffer_block_ptr[1], cur_block_size[1], copy_type, stream));
      offset += cur_block_size[1] / sizeof(float);
    }

    if (outType >= 0) {  // sync before return if outType >= 0
      cudaStreamSynchronize(stream);
    }

    if (nullptr != outData ) {
      if( !savePath.empty()) {
        saveOutBuf(outData, tot_out_size);
      }
      if (!constResultFile.empty()) { //nullptr != outData && 
        loadBuf(constResultFile, outData, tot_out_size, 0);
      }
    }

    return;  // release stream & buffes in DestoryEngine()
  }

  if (runTest > 0) {  // else: test time and check result(opt)
    int iterations = 3;
    int avgRuns = runTest;
    std::vector<float> times(avgRuns);

    int out_ulp = -1;  // NotSet;
    float *outRef = nullptr;
    if (debug_builder) {
      outRef = new float[tot_out_size];
      if (outType & 0x100) {  // bit 8, check output with reference data
        outType &= 0xFF;
        out_ulp = 1e4;  // Max error about 10-3. Most error < 10-5, a few ~= 6e-4
        memcpy(outRef, outData, tot_out_size);
      }
    }

    for (int j = 0; j < iterations; j++) {
      if (ifRunProfile && j == 1) {  // skip the first iteration
        context->setProfiler(&gProfiler);
        TIMING_ITERATIONS = (iterations - 1) * avgRuns;
      }

      float total = 0, ms;
      for (int i = 0; i < avgRuns; i++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (hasImplicitBatchDimension) {
          context->execute(batchSize, &buffers[0]);
        } else {
          context->executeV2(&buffers[0]);
        }

        int offset = 0;  // offset of OutData while batch > 1
        if (nullptr != outData) {
          {  // plan copy of all gpu output buffer
            CHECK_CUDA(cudaMemcpyAsync(outData, buffer_block_ptr[1], cur_block_size[1], cudaMemcpyDeviceToHost, stream));
            offset += cur_block_size[1] / sizeof(float);
          }
          /*
          // process variable buffer for fsd + keypoint
          if (1 == outType && fsd_out_num > 0) {
            std::vector<void *> curBuffers(fsd_out_num + 2);
            nvinfer1::Dims curBufferDims[ONNXTRT_MAX_BUFFERNUM];
            int idx_offset = nbBindings - fsd_out_num;  // skip AP out
            for (int i = 0; i < fsd_out_num; i++) {
              int ib = bufferIndex[i + idx_offset];
              curBuffers[i + 2] = buffers[ib];
              DPRINTF(2, "curBuffers[%d] = idx %d\n", i + 2, ib);
              curBufferDims[i + 2] = bufferDims[ib];
            }
            for (int nb = 0; nb < batchSize; nb++) {  // prcoess batch size <=32
              RunMaskNet(curBuffers, curBufferDims, outData + offset, buffers[inputIndex], stream, nb, batchSize);
              for (int i = 0; i < fsd_out_num; i++) {
                curBuffers[i + 2] += volume(curBufferDims[i + 2]) * sizeof(float);
              }
              offset += MaskBlockSize;
            }
          }
          */
        }
        cudaStreamSynchronize(stream);

        auto t_end = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

        if (debug_builder) {
          if (out_ulp < 0) {  // Not set yet, copy outData as reference
            memcpy(outRef, outData, tot_out_size);
            out_ulp = 4;  // Max error about 4.8x10-7
          } else {
            checkOutput((float*)outData, outRef, offset, out_ulp);
          }
        }

        times[i] = ms;
        total += ms;
      }
      total /= avgRuns;
      std::cout << "CH." << ch << " Average over " << avgRuns << " runs is " << total << " ms." << std::endl;

      ShowCudaMemInfo();
    }

    if (debug_builder) {
      for (int ib = 0; ib < nbBindings; ++ib) {
        int bindingIdx = bufferIndex[ib];
        for (int nb = 0; nb < batchSize; ++nb) {
          int64_t bufferSizesOutput = buffersSizes[bindingIdx];
          const char *sBufferType[ONNXTRT_MAX_BUFFERTYPE] = {"In:", "Out:", "Var:"};
          DPRINTF(1, "%s[%d]%s.%d buffeSize:%ld Data:\n", sBufferType[bufferInfo[bindingIdx].nBufferType], bindingIdx,
                  bufferInfo[bindingIdx].name, nb, bufferSizesOutput);
          printOutput<float>(bufferSizesOutput, buffers[bindingIdx] + nb * bufferSizesOutput);
        }
      }

      delete[] outRef;
    }

    if (ifRunProfile && TIMING_ITERATIONS > 0) {
      gProfiler.printLayerTimes();
    }

    if (nullptr != outData && !savePath.empty()) saveOutBuf(outData, tot_out_size);

    return;
  }
}

void IWorkspace::release() {
  status = -2;
  // Release the context, stream and the buffers
  for (int i = 0; i < ONNXTRT_MAX_BUFFERNUM; i++) {
    if (nullptr != mGraphs[i]) {
      delete mGraphs[i];
      mGraphs[i] = nullptr;
    }
  }
  if (nullptr != context) {
    cudaStreamDestroy(stream);
    context->destroy();
    context = nullptr;
  }
  if (nbBindings > 0) {
    CHECK_CUDA(cudaFree(bufferInfo[0].p));
    for (int i = 0; i < nbBindings; ++i) {
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

  if (gworkspace_second[ch].isInited()) {
    gworkspace_second[ch].release();
  }

  
  if (nullptr != _voxelGen) {
    _voxelGen -> terminate();
    delete _voxelGen;
    _voxelGen = nullptr;
  }

  if (nullptr != _gpuTimer) {
    printf("The latency of inference is %.5lf ms, with %d iterations.\n", _timeAccumulator, _iterationTimes);
    printf("The average speed of inference is %.5lf ms\n", _timeAccumulator/_iterationTimes);
    delete _gpuTimer;
    _gpuTimer = nullptr;
  }

  status = 0;
}

// interface for so/python
/*
 * ch: channel ID for multiple models:  0:maskrcnn, 1:resnet ...
 * engine_filename: tensorrt engine file name(<256B) or buffer
 * pMaskWeight: mask weight for maskrcnn/retinamask
 */
extern "C" int CreateEngine(int ch, const char *engine_filename, const char *config_string) {
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
  } else {
    return ONNXTRT_PARAMETER_ERR;
  }

  const unsigned short TRT_MAGIC[2] = {0x7470, 0x7472};  // TRT 6 & 7
  size_t size = 0;
  char *trtModelStream{nullptr};
  size_t name_len = strnlen(engine_filename, NAME_MAX + 1);
  if (name_len < NAME_MAX && name_len > 4 && 0 == strncasecmp(&engine_filename[name_len - 4], ".trt", 4)) {
    char full_filename[NAME_MAX];  // strtok will replace ','
    strncpy(full_filename, engine_filename, sizeof(full_filename));
    // check if has two trt file, split by ','
    char *pFirst = strtok(full_filename, ",");
    char *pSecond = strtok(NULL, ",");
    if (pFirst && pSecond) {
      size = ReadBinFile(pSecond, trtModelStream);
      if (0 < size) {
        DPRINTF(1, "Read Second trt: %-60.60s size=%ld\n", pSecond, size);
        gworkspace_second[ch].init(ch, trtModelStream, size, config_string);
        gworkspace_second[ch].setStatus(1);
        delete[] trtModelStream;
      }
    }
    size = ReadBinFile(pFirst, trtModelStream);
    if (0 == size) {
      DPRINTF(1, "Cannot open %-60.60s \n", full_filename);
      return ONNXTRT_IO_ERR;
    }
    DPRINTF(1, "Read First trt: %-60.60s size=%ld\n", full_filename, size);
  } else if (0 == memcmp(TRT_MAGIC, engine_filename, 4)) {
    trtModelStream = (char *)engine_filename;
    size = *(int *)(engine_filename + 12) + 0x18;  // TensorRT 6 format
    if (size < 1e3 || size > 1e9) {
      DPRINTF(1, "CreateEngine failed from %-60.60s \n", engine_filename);
      return ONNXTRT_IO_ERR;
    }
  } else {
    DPRINTF(1, "CreateEngine failed from %-60.60s \n", engine_filename);
    return ONNXTRT_IO_ERR;
  }

  int ret = gworkspace[ch].init(ch, trtModelStream, size, config_string);

  if (trtModelStream && trtModelStream != engine_filename) {
    delete[] trtModelStream;
  }

  if (ret < 0) {
    DPRINTF(1, "DeserializeCudaEngine failed from %-60.60s \n", engine_filename);
    return ret;
  }

  gworkspace[ch].setStatus(1);
  return ch;
}

// inputType: 0:cpu float32, 1:gpu float32, 2: gpu 3HW (CHW) float32 3: gpu 3HW (CHW) int8 4:lidar CPU pcd 
// outType: 0:cpu roi & feat map, 1: cpu roi & mask
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType, char *outData, int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  gworkspace[ch].setStatus(-2);
  
  if(gworkspace[ch]._isGtInitialized && gworkspace[ch]._warmedUp) {
    gworkspace[ch]._gpuTimer->Start();
    gworkspace[ch].doInference(inputData, outData, batch, 0, inputType, outType);
    gworkspace[ch]._gpuTimer->Stop();
    if(outType >= 0)  gworkspace[ch]._iterationTimes  += 1;
    
    gworkspace[ch]._timeAccumulator += gworkspace[ch]._gpuTimer->Elapsed();
    
  } else if(gworkspace[ch]._isGtInitialized && !gworkspace[ch]._warmedUp) {
    gworkspace[ch].doInference(inputData, outData, batch, 0, inputType, outType);
    gworkspace[ch]._warmedUp = true;
  }else {
    gworkspace[ch].doInference(inputData, outData, batch, 0, inputType, outType);
  }
    
  gworkspace[ch].setStatus(1);
  return ONNXTRT_OK;
}

extern "C" int DestoryEngine(int ch) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  gworkspace[ch].release();

  gworkspace[ch].setStatus(-2);
  //DestroyMaskNet();
  gworkspace[ch].setStatus(0);

  return ONNXTRT_OK;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
  return gworkspace[ch].getBufferInfo(pBufferInfo, pNum, pStream);
}

int RunProfile(int ch, int batch, char *inputData, int inputType, char *outData, int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }

  gworkspace[ch].setStatus(-2);
  gworkspace[ch].doInference(inputData, outData, batch, 5, inputType, outType);
  gworkspace[ch].setStatus(1);
  return ONNXTRT_OK;
}

//These two functions are for test multi model
extern "C" void* AllocateSpaceGPU(size_t bytes, int num) {
  void* gpuTmpPtr = nullptr;
  CHECK_CUDA(cudaMalloc(&gpuTmpPtr, bytes * num));

  return gpuTmpPtr;
}

extern "C" void MemcpyHost2DeviceGPU(void* dst, void* src, size_t bytes, int num) {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));
  CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes * num, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}

extern "C" int ParseEngineData(void *input, void *output, int batchSize, int outType) { return -1; }
int TRT_DEBUGLEVEL=1;
// static funtion called when using dlopen & dlclose
static __attribute__((constructor)) void lib_init(void) {
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
    }
  }
  DPRINTF(1, "Load onnx2trt lib %s built@%s %s DebugLevel=%d\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__,
          TRT_DEBUGLEVEL);
}

static __attribute__((destructor)) void lib_deinit(void) {
  DPRINTF(1, "Unload onnx2trt lib %s built@%s %s\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__);
}
