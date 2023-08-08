/*
 * Copyright (c) 2018, Xpeng Motors. All rights reserved.
 *
 */

#pragma once

#include <assert.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <vector>
#include <queue>
#include <set>

#include <ctype.h>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>

#include <cuda.h>

//#include "common.h"
#include "NvInfer.h"
#include "NvUffParser.h"
using namespace nvinfer1;
using namespace nvuffparser;

extern "C" int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH <= 1
#define FIX_BUG_TENOSRRT5 1  // just for tensorrt5.0.0.x, must be 0 for tensorrt4.x or tensorrt5.0.2.x
#endif

extern "C" int filteMOD(int batch, const void* inputbuf, void* outputbuf, const nvinfer1::Dims dim, float score_thresh,
                        void* workspace, size_t ws_size, cudaStream_t stream);

extern "C" int SetConvLSTMState(cudaStream_t stream, int state);

// Refer from sampleUffSSD.cpp
inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
#if NV_TENSORRT_MAJOR >= 7
    case nvinfer1::DataType::kBOOL:
#endif
    case nvinfer1::DataType::kINT8:
      return 1;
#if NV_TENSORRT_MAJOR >= 4
    case nvinfer1::DataType::kINT32:
      return 4;
#endif
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

#ifndef CHECK_CUDA
#define CHECK_CUDA(status)                               \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      fprintf(stderr, "Cuda CHECK error at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(ret)); \
      fflush(stderr);                                    \
    }                                                    \
  } while (0)
#endif

#define cudaCheck(status) CHECK_CUDA(status)

#define CUDA_ERROR_CHECK
#define cudaEngineCheck(ch, status) __cudaEngineCheck(ch, status, __FILE__, __LINE__)

inline void __cudaEngineCheck(const int ch, const int status, const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaEngineCheck(CH=%d,Status=%d) failed at %s:%i : %s\n", 
            ch, status, file, line, cudaGetErrorString(err));
    fflush(stderr); //exit(-1);
  }
#endif

  return;
}

//! Refer from: tensorrt/samples/common/sampleDevice.h
//! \class TrtCudaGraph
//! \brief Managed CUDA graph
//!
class TrtCudaGraph {
public:
  explicit TrtCudaGraph() = default;

  TrtCudaGraph(const TrtCudaGraph&) = delete;

  TrtCudaGraph& operator=(const TrtCudaGraph&) = delete;

  TrtCudaGraph(TrtCudaGraph&&) = delete;

  TrtCudaGraph& operator=(TrtCudaGraph&&) = delete;

  ~TrtCudaGraph();

  void beginCapture(cudaStream_t& stream);

  void launch(cudaStream_t& stream);

  void endCapture(cudaStream_t& stream);

  bool deleteNodes(std::vector<int> id);

  bool destroyOneNode(int i);

  void updateGraphExec();

  void updateNodesAndEdges();

  void printGraph();

private:
  cudaGraph_t mGraph{};
  cudaGraphExec_t mGraphExec{};

  size_t numNodes = 0;
  std::vector<cudaGraphNode_t> nodes;

  size_t numEdges = 0;
  std::vector<cudaGraphNode_t> from;
  std::vector<cudaGraphNode_t> to;

  cudaError error;
};

/* This struct deals with the CUDA kernel */
struct kernel {
  uint32_t v0;
  uint32_t v1;
  uint32_t v2;
  uint64_t v3;
  uint32_t v4;
  uint32_t v5;
  uint32_t v6;
  uint32_t v7;
  uint32_t v8;
  void *module;
  uint32_t size;
  uint32_t v9;
  void *p1;   
};

struct dummy1 {
  void *p0;
  void *p1;
  uint64_t v0;
  uint64_t v1;
  void *p2;
};

/* The function struct!!! */
struct CUfunc_st {
  uint32_t v0;
  uint32_t v1;
  char *name;
  // std::string name;
  uint32_t v2;
  uint32_t v3;
  uint32_t v4;
  uint32_t v5;
  struct kernel *kernel;
  void *p1;
  void *p2;
  uint32_t v6;
  uint32_t v7;
  uint32_t v8;
  uint32_t v9;
  uint32_t v10;
  uint32_t v11;
  uint32_t v12;
  uint32_t v13;
  uint32_t v14;
  uint32_t v15;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;
  uint32_t v20;
  uint32_t v21;
  uint32_t v22;
  uint32_t v23;
  struct dummy1 *p3;
};

struct myf {
  uint64_t addrs[256];
};

void Dump(const void * mem, unsigned int n);

bool isBadPtr(const char* p);

void printFuncInfo(struct CUfunc_st *pFunc, std::string printHead, bool if_dump, bool if_ptr, bool if_value);
