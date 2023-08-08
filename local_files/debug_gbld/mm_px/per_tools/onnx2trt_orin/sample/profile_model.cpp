/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * [caizw@20181130] Add openCV to read encoded images( readImageByOpenCV ) ,
 * [caizw@20210510] Add stb_image to read encoded images( png/jpg/ppm ) , 
 * openmp to run  mulit-models parallel example:
   bin/maskrcnn_mulit_models -s bin/libonnxtrt.so
    -e model_final_608x960b1.trt -i output6_960x608_134544.645.ppm
    -o output_final_960x608_mulit.ppm -m masknetweight.bin
    -e LLD_604x960fp32_MKZ.trt -i input_960_604.ppm
    -o output_LLD_960x604_mulit.ppm
 * MaskNet need weight "masknetweight.bin"
 */

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>  // For ::getopt
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
#include <fcntl.h>  // For ::open
#include <float.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>
#include <map>
#include "onnxtrt.h"
#include "NvInferPlugin.h"

PTCreateEngine pCreateEngine;
PTRunEngine pRunEngine;
PTDestoryEngine pDestoryEngine;
PTGetBufferOfEngine pGetBuffer;

#include <cuda_runtime_api.h>

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

// return ms
double getTimeDiff(struct timeval *tm) {
  struct timeval lasttm = *tm;
  gettimeofday(tm, NULL);
  return (tm->tv_sec - lasttm.tv_sec) * 1000.f + (tm->tv_usec - lasttm.tv_usec) / 1000.f;
}

int getPreprocesParams(int &modelType, int &im_w, int &im_h, int *idxRGB, float *Scales, float *Means) {
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
  if (3 <= modelType) {
    Scales[0] = 255.f;  // R
    Scales[1] = 255.f;  // G
    Scales[2] = 255.f;  // B
  }

  return 0;
}

//97MB 519ms, 74MB 399ms
size_t ReadBinFileC(const char *filename, char *&databuffer) {
#if 0	
  struct stat f_stat;
  if (stat(filename, &f_stat) == -1)
  {
      return -1;
  } 
  size_t size = f_stat.st_size;
#endif			
  size_t size{0};
  FILE *fp = fopen(filename, "rb");
  if (fp) {  
    fseek(fp, 0, SEEK_END);
    size=ftell (fp); 
    fseek(fp, 0, SEEK_SET);

    databuffer = new char[size];
    // assert(databuffer);
    size = fread(databuffer, 1, size, fp);
    fclose(fp);
  }
  return size;
}

// 97MB: 524ms, 402ms
size_t ReadBinFile(const char *filename, char *&databuffer) {
  size_t size{0};
  std::ifstream file(filename, std::ifstream::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    databuffer = new char[size];
    // assert(databuffer);
    file.read(databuffer, size);
    file.close();
  }
  return size;
}

//stb support ppm/pgm, jpeg, png
int readImageBySTB(const char *filename, unsigned char *&rgbbuffer, int &im_w, int &im_h, char *&databuffer,
                   int modelType = 1) {
  return 0;
}

#ifndef USE_CV
int saveImageByOpenCV(std::string filename, unsigned char *imgbuffer, int im_w, int im_h) { return -1; }
#else
#include <opencv2/opencv.hpp>
int saveImageByOpenCV(std::string filename, unsigned char *imgbuffer, int im_w, int im_h) {
  cv::Mat bgrImg(im_h, im_w, CV_8UC3, imgbuffer);
  cv::Mat rgbImg(im_h, im_w, CV_8UC3);

  int from_to[] = {0, 2, 1, 1, 2, 0};  // From bgr to rgb
  cv::mixChannels(&bgrImg, 1, &rgbImg, 1, from_to, 3);

  cv::imwrite(filename, rgbImg);

  return 0;
}
#endif

void print_usage() {
  cout << "TensorRT maskcnn sample" << endl;
  cout << "Usage: onnx2trt onnx_model.pb"
      << "\n\t[-s libonnxtrt.so] (path of libonnxtrt.so)"
      << "\n\t[-e engine_file.trt  (test TensorRT engines, multiple times)"
      << "\n\t[-i input_data.bin]  (input datas)"
      << "\n\t[-m modelConfig ( ex: Task=LLD|MOD|KPTL,Prioity=High,CudaGraph=True) ]"
      << "\n\t[-b max_batch_size (default=1)]"
      << "\n\t[-B min_batch_size (default = max_batch_size)]"       
      << "\n\t[-o output.ppm]  (output image as ppm) ]"       
      << "\n\t[-t test_number] (test iteration number, default=100)"    
      << "\n\t[-F frequency] (frequency of inference.)"           
      << "\n\t[-h] (show help)" << endl;
}

float percentile(float percentage, std::vector<float> &times, float total_ms) {
  int all = static_cast<int>(times.size());
  int exclude = static_cast<int>((1 - percentage / 100) * all);
  if (0 <= exclude && exclude <= all) {
    std::sort(times.begin(), times.end());
    float pctTime = times[all == exclude ? 0 : all - 1 - exclude];
    float totTime = 0;
    for (int i = 5; i < all - 5; i++) totTime += times[i];  // drop the fist & last 5 datas.

    printf("TestAll %d in %.1fms range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< %.3fms\n",
        all, total_ms, times[0], times[all - 1], totTime / (all - 10), times[all / 2], percentage, pctTime);
    return pctTime;
  }
  return std::numeric_limits<float>::infinity();
}

static int testRunEngine(int ch, int engine_chID, int batch, char *inputData, int inputType, char *outData, int outType) {
  if (NULL == pRunEngine) return -1;

  auto t_start = std::chrono::high_resolution_clock::now();

  int nRet = pRunEngine(engine_chID, batch, inputData, inputType, outData, outType);

  float ms = std::chrono::duration<float, std::milli>(  // 0.4us
                std::chrono::high_resolution_clock::now() - t_start)
                .count();
  printf("CH[%d] time = %f\n", ch, ms);  // cout : 0.1ms, printf: 0.03ms

  return nRet;
}

void checkShort(short *outRef, short* outData, int size, int ulp=4) {
      
  int count_notEq = 0;
  int eltCount = size / sizeof(short);
  for (int i = 0; i< eltCount; i++) {
    if (std::abs(outData[i] - outRef[i]) >= ulp) {
      printf("%d Not Equal: %d vs %d\n", i, outData[i], outRef[i]);
      if (++count_notEq >= 16) break;
    }
  }
  printf("checkOutput count_notEq = %d\n", count_notEq );
}

// Check output if unstable
void checkOutput(int ch, int outType, char* outptr, int size) {
static char* outRefs[16] = {nullptr};

  if (NULL == outptr) {
    return;
  } else if (0 == size && outRefs[ch]) { // last run, free
    delete[] outRefs[ch];
    outRefs[ch] = NULL;
    return;
  } else if (NULL == outRefs[ch]){ // first run, alloc and copy
    outRefs[ch] = new char[size];
    memcpy(outRefs[ch], outptr, size);
    return;
  } 
  printf("checkOutput outType = %d\n", outType ); 
  if (9 == outType){
    checkShort((short*)outRefs[ch], (short*)outptr, size);
  }
}

#include <condition_variable>
#include <mutex>
static std::mutex mtx;
static std::condition_variable condvar;
bool ready = true;
float g_times[16], max_ms;
std::chrono::time_point<std::chrono::high_resolution_clock> g_start;
static int testRunEngine1(int ch, int engine_chID, int batch, char *inputData, int inputType, int frame_num, char *outData,
                          int outType) {
  int nRet = 0;
  int nT = 0;
  {
    std::unique_lock<std::mutex> lck(mtx);
    condvar.wait(lck);
  }
  while (ready) {
    int inT = inputType + (((nT++) % frame_num) ? 2 : 0);
    nRet = pRunEngine(engine_chID, batch, inputData, inT, outData, outType);

    g_times[ch] = std::chrono::duration<float, std::milli>(  // 0.4us
                  std::chrono::high_resolution_clock::now() - g_start).count();
    printf("CH[%d] time = %f\n", ch, g_times[ch]);  // cout : 0.1ms, printf: 0.03ms
    max_ms = g_times[ch];
    {
      std::unique_lock<std::mutex> lck(mtx);
      condvar.wait(lck);
    }
  };

  return nRet;
}

// print the fist & last 64(PrintNum) data
template <typename T>
void printOutput(int64_t eltCount, T *outputs, int bits = 32) {
  const int PrintNum = 64;
  for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {
    std::cerr << outputs[eltIdx] << "\t, ";
  }
  if (eltCount > PrintNum) {
    std::cerr << " ... ";
    for (int64_t eltIdx = (eltCount - PrintNum); eltIdx < eltCount; ++eltIdx) {
      std::cerr << outputs[eltIdx] << "\t, ";
    }
  }

  std::cerr << std::endl;
}

//Print the value of filter
void printFilter(int nDims, int d[], float *outputs) {
  const int PrintNum = 16;
  int eltCount = d[1];
  for( int i = 0; i< d[0]; i++) {
    int index = ((int*)outputs)[0];
    if( index < 0 ) {
      break;
    }
    std::cerr << i << ": Idx."<< index << "\t: ";
    for (int eltIdx = 1; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {
        std::cerr << outputs[eltIdx] << "\t, ";
    }
    if (eltCount > PrintNum) {
      if (eltCount > 2*PrintNum) {
        std::cerr << " ... ";
      }
      for (int64_t eltIdx = std::max(PrintNum, (eltCount - PrintNum)); eltIdx < eltCount; ++eltIdx) {
        std::cerr << outputs[eltIdx] << "\t, ";
      }
    }
    std::cerr << std::endl;
    outputs += d[1];
  }
}

// print the data of buffer, split by bufferInfo
void PrintOutBuffer(float *pBuf, int outType, int batchsize, int bufferNum, EngineBuffer *bufferInfo) {
  int valNum = 0;

  if (2 == outType) {  // packed buffers by batchsize
    for (int nb = 0; nb < batchsize; nb++) {
      for (int i = 0; i < bufferNum; i++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++) eltCount *= bufferInfo[i].d[d];
        printf("Out[%d].%d eltCount:%d Data:\n", i, nb, eltCount);
        printOutput(eltCount, pBuf + valNum);
        valNum += eltCount;
      }
    }
  } else if (8 == outType || 9 == outType) {  // for int8 or fp16
    for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = bufferInfo[i].nBufferSize / 2;
        if (9 == outType)  // fp16 as short
          printOutput(eltCount, (unsigned short *)bufferInfo[i].p, 16);
        else
          printOutput(eltCount, (unsigned char *)bufferInfo[i].p, 8);
      }
    }
  } else  {  // plan buffers, default
    for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++) eltCount *= bufferInfo[i].d[d];
        printf("[%d]%s.%d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
        if( strstr(bufferInfo[i].name, "_filter")!=NULL){
          printFilter(bufferInfo[i].nDims, bufferInfo[i].d, pBuf + valNum);
        } else {
          printOutput(eltCount, pBuf + valNum);
        }
        valNum += eltCount;
      }
    }
  }
}

// convert 3 channel to 12 channel
int res2channel(float *pSrc, int im_w, int im_h) {
  if( nullptr == pSrc || 0 == im_w || 0 == im_h ){
    return -1;
  }  

  printf("res2channel %dx%d\n", im_w, im_h);
  int im_size = 3*im_w*im_h;
  float *pDst = new float[im_size];
  int src_idx=0;
  for (int c = 0; c < 3; c++) {
    for (int y = 0; y < im_h; y++) {
      for (int x = 0; x < im_w; x++) {
        int dst_c = ((x & 1) * 2 + (y & 1)) * 3 + c;
        int dst_y = y / 2;
        int dst_x = x / 2;
        int dst_idx = dst_c * (im_w / 2) * (im_h / 2) + dst_y * (im_w / 2) + dst_x;    

        pDst[dst_idx] = pSrc[src_idx ++];
      }
    }
  }
  memcpy( pSrc, pDst, im_size * sizeof(float) );
  delete[] pDst;

  return 0;
}

// convert 2channel float-chw to 12channel fp16-chw16
int transform_fp16_chw16(float *pSrc, int im_w, int im_h) {
  if (nullptr == pSrc || 0 == im_w || 0 == im_h) {
    return -1;
  }  

  return 0;  
}

typedef struct IMGInfo {
  int im_w = 0;
  int im_h = 0;
  unsigned char *rgbbuffer = NULL;
  char *inputStream;
  int modelType;  // 0: LLD, 1: MaskRCNN/FasterRCNN, 8: Byte output
} IMGInfo;
 
inline float sigmoid(float x) { return (1 / (1 + expf(-x))); }

#include <cuda.h>
#include <cupti.h>

// Standard NVTX headers
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"

// Includes definition of the callback structures to use for NVTX with CUPTI
#include "generated_nvtx_meta.h"

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;

typedef struct {
  const char *funcName;
  uint32_t correlationId;
} ApiData;

typedef std::map<uint64_t, ApiData> nodeIdApiDataMap;
nodeIdApiDataMap nodeIdCorrelationMap;

static const char * getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    default:
      break;
  }

  return "<unknown>";
}

const char * getName(const char *name) {
  if (name == NULL) {
      return "<null>";
  }
  return name;
}

const char * getDomainName(const char *name) {
  if (name == NULL) {
      return "<default domain>";
  }
  return name;
}

const char * getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
        return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD:
        return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
        return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM:
        return "STREAM";
    default:
        break;
  }

  return "<unknown>";
}

uint32_t getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id) {
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
        return id->pt.processId;
    case CUPTI_ACTIVITY_OBJECT_THREAD:
        return id->pt.threadId;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
        return id->dcs.deviceId;
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        return id->dcs.contextId;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
        return id->dcs.streamId;
    default:
        break;
  }

  return 0xffffffff;
}

static void printActivity(CUpti_Activity *record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      CUpti_ActivityMemcpy4 *memcpy = (CUpti_ActivityMemcpy4 *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, size %llu, correlation %u, graph ID %u, graph node ID %llu\n",
              getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
              (unsigned long long) (memcpy->start - startTimestamp),
              (unsigned long long) (memcpy->end - startTimestamp),
              memcpy->deviceId, memcpy->contextId, memcpy->streamId,
              (unsigned long long)memcpy->bytes, memcpy->correlationId,
              memcpy->graphId, (unsigned long long)memcpy->graphNodeId);

      // Retrieve the information of the API used to create the node
      nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(memcpy->graphNodeId);
      if (it != nodeIdCorrelationMap.end()) {
          printf("Graph node was created using API %s with correlationId %u\n", it->second.funcName, it->second.correlationId);
      }
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
      CUpti_ActivityMemset3 *memset = (CUpti_ActivityMemset3 *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u, graph ID %u, graph node ID %llu\n",
             memset->value,
             (unsigned long long) (memset->start - startTimestamp),
             (unsigned long long) (memset->end - startTimestamp),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId, memset->graphId, (unsigned long long)memset->graphNodeId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel6 *kernel = (CUpti_ActivityKernel6 *) record;
      printf("%s \"%s\" %llu [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name, (unsigned long long) (kernel->end - kernel->start), 
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u), graph ID %u, graph node ID %llu\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory,
             kernel->graphId, (unsigned long long)kernel->graphNodeId);

      // Retrieve the information of the API used to create the node
      nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(kernel->graphNodeId);
      if (it != nodeIdCorrelationMap.end()) {
          printf("Graph node was created using API %s with correlationId %u\n", it->second.funcName, it->second.correlationId);
      }

      break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME: {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      const char* apiName;
      CUPTI_CALL(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &apiName));
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u, Name %s\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId, apiName);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MARKER: {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER  id %u [ %llu ], name %s, domain %s\n",
              marker->id, (unsigned long long) marker->timestamp, getName(marker->name), getDomainName(marker->domain));
      break;
    }
    case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER  id %d, color %d, category %d\n", marker->id, marker->color, marker->category);
      break;
    }   
    default: {
      printf("  <unknown> record->kind = %d \n", record->kind);
      break;
    }
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  static const char* funcName;
  static const char* symbolName;
  static char g_symbolName[1024];
  static uint32_t correlationId;
  // static const cudaLaunchKernel_ptsz_v7000_params* ownKernelParams;
  // static const cudaLaunchKernel_v7000_params* trtKernelParams;
  // static const cudaMemcpyAsync_ptsz_v7000_params_st* memcpyAsyncParams;
  // static const cudaMemcpy2DAsync_v3020_params_st* memcpy2DAsyncParams;
  // static const cudaMemcpy2DAsync_ptsz_v7000_params_st* memcpy2DAsyncPtszParams;
  // static const cudaMemsetAsync_ptsz_v7000_params_st* memsetAsyncParams;

  // Check last error
  //CUPTI_CALL(cuptiGetLastError());

  switch (domain) {
    case CUPTI_CB_DOMAIN_RESOURCE: {
      CUpti_ResourceData *resourceData = (CUpti_ResourceData *)cbInfo;
      switch (cbid) {
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED: {
          // Do not store info for the nodes that are created during graph instantiate
          if (!strncmp(funcName, "cudaGraphInstantiate", strlen("cudaGraphInstantiate"))) {
            break;
          }
          CUpti_GraphData *cbData = (CUpti_GraphData *) resourceData->resourceDescriptor;
          uint64_t nodeId;

          // Query the graph node ID and store the API correlation id and function name
          CUPTI_CALL(cuptiGetGraphNodeId(cbData->node, &nodeId));
          ApiData apiData;
          apiData.correlationId = correlationId;
          apiData.funcName = funcName;
          nodeIdCorrelationMap[nodeId] = apiData;
          if (g_symbolName[0] == 0) {
            printf("GraphNode Create\t%lu\t%s\n", nodeId, funcName);
          } else {
            printf("GraphNode Create\t%lu\t%s\t%s\n", nodeId, funcName, g_symbolName);
          }
          break;
        }
        case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED: {
          CUpti_GraphData *cbData = (CUpti_GraphData *) resourceData->resourceDescriptor;
          uint64_t nodeId, originalNodeId;

          // Overwrite the map entry with node ID of the cloned graph node
          CUPTI_CALL(cuptiGetGraphNodeId(cbData->originalNode, &originalNodeId));
          nodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(originalNodeId);
          if (it != nodeIdCorrelationMap.end()) {
            CUPTI_CALL(cuptiGetGraphNodeId(cbData->node, &nodeId));
            ApiData apiData = it->second;
            nodeIdCorrelationMap.erase(it);
            nodeIdCorrelationMap[nodeId] = apiData;
          }
          printf("GraphNode Cloned %lu\n", originalNodeId);
          break;
        }
        default: {
          break;
        }
      }
    }
    break;
    case CUPTI_CB_DOMAIN_DRIVER_API: {
      if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        correlationId = cbInfo->correlationId;
        funcName = cbInfo->functionName;
        g_symbolName[0] = 0;
        // printf("Driver API captured:\t%s\n", funcName);
        switch (cbid)
        {
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: // 307
          symbolName = cbInfo->symbolName;
            if(isBadPtr(symbolName)) {
              break;
            }
            strncpy(g_symbolName, symbolName, sizeof(g_symbolName) - 1);
            printf("Driver API captured:\t%s\t%s\n", funcName, g_symbolName);
          break;
        case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz: // 442
          symbolName = cbInfo->symbolName;
            if(isBadPtr(symbolName)) {
              break;
            }
            strncpy(g_symbolName, symbolName, sizeof(g_symbolName) - 1);
            printf("Driver API captured:\t%s\t%s\n", funcName, g_symbolName);
          break;
        default:
          // printf("Driver API captured:\t%s\n", funcName);
          break;
        }
      }
      break;
    }
    case CUPTI_CB_DOMAIN_RUNTIME_API: {
      if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        correlationId = cbInfo->correlationId;
        funcName = cbInfo->functionName;
        g_symbolName[0] = 0;
        // printf("API captured:\t%s\n", funcName);
        switch (cbid) {
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000: // 214
            symbolName = cbInfo->symbolName;
            if(isBadPtr(symbolName)) {
              break;
            }
            strncpy(g_symbolName, symbolName, sizeof(g_symbolName) - 1);
            printf("Runtime API captured:\t%s\t%s\n", funcName, g_symbolName);

            // ownKernelParams = (cudaLaunchKernel_ptsz_v7000_params*)cbInfo->functionParams;
            // printf("Kernel func %p\n", ownKernelParams->func);
            // printf("Kernel gridDim %d %d %d\n", ownKernelParams->gridDim.x, ownKernelParams->gridDim.y, ownKernelParams->gridDim.z);
            // printf("Kernel blockDim %d %d %d\n", ownKernelParams->blockDim.x, ownKernelParams->blockDim.y, ownKernelParams->blockDim.z);
            // printf("Kernel args %p\n", &ownKernelParams->args);
            // printf("Kernel sharedMem %d\n", ownKernelParams->sharedMem);
            // printf("Kernel stream %p\n", &ownKernelParams->stream);
          break;
          case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: // 211
            symbolName = cbInfo->symbolName;
            if(isBadPtr(symbolName)) {
              break;
            }
            strncpy(g_symbolName, symbolName, sizeof(g_symbolName) - 1);
            printf("Runtime API captured:\t%s\t%s\n", funcName, g_symbolName);

            // trtKernelParams = (cudaLaunchKernel_v7000_params*)cbInfo->functionParams;
            // printf("Kernel func %p\n", trtKernelParams->func);
            // printf("Kernel gridDim %d %d %d\n", trtKernelParams->gridDim.x, trtKernelParams->gridDim.y, trtKernelParams->gridDim.z);
            // printf("Kernel blockDim %d %d %d\n", trtKernelParams->blockDim.x, trtKernelParams->blockDim.y, trtKernelParams->blockDim.z);
            // printf("Kernel args %p\n", &trtKernelParams->args);
            // printf("Kernel sharedMem %d\n", trtKernelParams->sharedMem);
            // printf("Kernel stream %p\n", &trtKernelParams->stream);
          break;
          case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_ptsz_v7000: // 225
            printf("Runtime API captured:\t%s\n", funcName);

            // memcpyAsyncParams = (cudaMemcpyAsync_ptsz_v7000_params_st*)cbInfo->functionParams;
            // printf("Memcpy dst %p\n", memcpyAsyncParams->dst);
            // printf("Memcpy src %p\n", memcpyAsyncParams->src);
            // printf("Memcpy size %d\n", memcpyAsyncParams->count);
            // printf("Memcpy kind %d\n", memcpyAsyncParams->kind);
            // printf("Memcpy stream %p\n", &memcpyAsyncParams->stream);
          break;
          case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020: // 44
            printf("Runtime API captured:\t%s\n", funcName);
            
            // memcpy2DAsyncParams = (cudaMemcpy2DAsync_v3020_params_st*)cbInfo->functionParams;
            // printf("Memcpy2D dst %p\n", memcpy2DAsyncParams->dst);
            // printf("Memcpy2D src %p\n", memcpy2DAsyncParams->src);
            // printf("Memcpy2D dpitch %d\n", memcpy2DAsyncParams->dpitch);
            // printf("Memcpy2D spitch %d\n", memcpy2DAsyncParams->spitch);
            // printf("Memcpy2D width %d\n", memcpy2DAsyncParams->width);
            // printf("Memcpy2D height %d\n", memcpy2DAsyncParams->height);
            // printf("Memcpy2D kind %d\n", memcpy2DAsyncParams->kind);
            // printf("Memcpy2D stream %p\n", &memcpy2DAsyncParams->stream);
          break;
          case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000: // 228
            printf("Runtime API captured:\t%s\n", funcName);
            
            // memcpy2DAsyncPtszParams = (cudaMemcpy2DAsync_ptsz_v7000_params_st*)cbInfo->functionParams;
            // printf("Memcpy2D dst %p\n", memcpy2DAsyncPtszParams->dst);
            // printf("Memcpy2D src %p\n", memcpy2DAsyncPtszParams->src);
            // printf("Memcpy2D dpitch %d\n", memcpy2DAsyncPtszParams->dpitch);
            // printf("Memcpy2D spitch %d\n", memcpy2DAsyncPtszParams->spitch);
            // printf("Memcpy2D width %d\n", memcpy2DAsyncPtszParams->width);
            // printf("Memcpy2D height %d\n", memcpy2DAsyncPtszParams->height);
            // printf("Memcpy2D kind %d\n", memcpy2DAsyncPtszParams->kind);
            // printf("Memcpy2D stream %p\n", &memcpy2DAsyncPtszParams->stream);
          break;
          case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_ptsz_v7000: // 235
            printf("Runtime API captured:\t%s\n", funcName);
            
            // memsetAsyncParams = (cudaMemsetAsync_ptsz_v7000_params_st*)cbInfo->functionParams;
            // printf("Memset devPtr %p\n", memsetAsyncParams->devPtr);
            // printf("Memset value %d\n", memsetAsyncParams->value);
            // printf("Memset count %d\n", memsetAsyncParams->count);
            // printf("Memset stream %p\n", &memsetAsyncParams->stream);
          break;
          default: {
            printf("Runtime API captured:\t%s\n", funcName);
            break;
          }
        }
        // switch (cbid) {
        //   case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000: // 214
        //   case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: // 211
        //     symbolName = cbInfo->symbolName;
        //     if(isBadPtr(symbolName)) {
        //       break;
        //     }
        //     strncpy(g_symbolName, symbolName, sizeof(g_symbolName) - 1);
        //   break;
        // }
      }
      break;
    }
    case CUPTI_CB_DOMAIN_NVTX:
    {
      CUpti_NvtxData* data = (CUpti_NvtxData*)cbInfo;
      switch (cbid) {
        case CUPTI_CBID_NVTX_nvtxDomainCreateA: {
          nvtxDomainCreateA_params* params = (nvtxDomainCreateA_params*)data->functionParams;
          break;
        }
        case CUPTI_CBID_NVTX_nvtxMarkEx: {
          nvtxMarkEx_params* params = (nvtxMarkEx_params*)data->functionParams;
          break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainMarkEx: {
          nvtxDomainMarkEx_params* params = (nvtxDomainMarkEx_params*)data->functionParams;
          break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
          nvtxDomainRangePushEx_params* params = (nvtxDomainRangePushEx_params*)data->functionParams;
          // auto name = params->core.eventAttrib->message.ascii;
          auto name = params->core.eventAttrib->message.unicode;
          printf("#CUPTI_CBID_NVTX_nvtxDomainRangePushEx\tname = %s\n", name);
          break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePop: {
          nvtxDomainRangePop_params* params = (nvtxDomainRangePop_params*)data->functionParams;
          printf("#CUPTI_CBID_NVTX_nvtxDomainRangePop\tdomain = %p\n", params->domain);
          break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRegisterStringA: {
          nvtxDomainRegisterStringA_params* params = (nvtxDomainRegisterStringA_params*)data->functionParams;
          printf("#CUPTI_CBID_NVTX_nvtxDomainRegisterStringA\tdomain = %p\n", params->domain);
          break;
        }
        // Add more NVTX callbacks, refer "generated_nvtx_meta.h" for all NVTX callbacks
        default: {
          printf("##########CUPTI_CB_DOMAIN_NVTX cbid = %u\n", cbid);
          break;
        }
      }
    }    
    break;
  
    default: {
      printf("##########CUPTI_CB_DOMAIN domain = %d\n", domain);
      break;
    }
  }
}

#define USE_CALLBACK 1
// #define USE_ACTIVITY 1 // coredump on orin
 
void initTrace() {
#ifdef USE_ACTIVITY
  // Enable activity record kinds.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  
  // For NVTX markers (Marker, Domain, Start/End ranges, Push/Pop ranges, Registered Strings)
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
  
  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
#endif
  
#ifdef USE_CALLBACK  
  CUpti_SubscriberHandle subscriber;
  CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler , NULL));
  // Enable callbacks for CUDA graph
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED));
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE));
#endif

  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace() {
#ifdef USE_ACTIVITY
   // Force flush any remaining activity buffers before termination of the application
   CUPTI_CALL(cuptiActivityFlushAll(1));
#endif   
}

#define DRIVER_API_CALL(apiFuncCall)                                       \
do {                                                                       \
  CUresult _status = apiFuncCall;                                          \
  if (_status != CUDA_SUCCESS) {                                           \
    const char* errstr;                                                    \
    cuGetErrorString(_status, &errstr);                                    \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, errstr);                     \
    exit(-1);                                                              \
  }                                                                        \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                      \
do {                                                                       \
  cudaError_t _status = apiFuncCall;                                       \
  if (_status != cudaSuccess) {                                            \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
            __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
    exit(-1);                                                              \
  }                                                                        \
} while (0)

#define CUPTI_CALL(call)                                                  \
do {                                                                      \
  CUptiResult _status = call;                                             \
  if (_status != CUPTI_SUCCESS) {                                         \
    const char *errstr;                                                   \
    cuptiGetResultString(_status, &errstr);                               \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
            __FILE__, __LINE__, #call, errstr);                           \
    exit(-1);                                                             \
  }                                                                       \
} while (0)

int main(int argc, char *argv[]) {
  // initialize the activity trace
  initTrace();

  if (argc <= 1) print_usage();
  std::vector<std::string> engine_filenames;
  std::vector<std::string> maskweight_filenames;
  std::vector<std::string> input_filenames;
  std::vector<std::string> out_filenames;
  std::vector<int> batch_sizes, min_batch;
  std::vector<int> engine_chID;
  float fps_delay = 10;  // delay ms to get fixed FPS
  const char *pSoName = "./libonnxtrt.so";
  bool debug_output = false;
  int MAX_TEST = 100;
  int thread_type = 2; // // Default thread type = 2, mulit-stream in one thread
  int video_frame_num = 1;  // 1:single image, >1: frame number of video

  int arg = 0;
  while ((arg = ::getopt(argc, argv, "o:e:c:i:m:s:b:B:F:t:T:v:gh")) != -1) {
    if ('o' == arg || 'e' == arg || 'i' == arg || 'm' == arg || 's' == arg || 'b' == arg || 'F' == arg || 't' == arg ||
        'T' == arg || 'c' == arg || 'v' == arg || 'B' == arg) {
      if (!optarg) {
        cerr << "ERROR: -" << arg << " flag requires argument" << endl;
        return -1;
      }
    }
    switch (arg) {
      case 'o':
        out_filenames.push_back(optarg);
        break;
      case 'e':
        engine_filenames.push_back(optarg);
        break;
      case 'c':
        engine_chID.push_back(atoi(optarg));
        break;        
      case 'i':
        input_filenames.push_back(optarg);
        break;
      case 'm':
        maskweight_filenames.push_back(optarg);
        break;
      case 's':
        pSoName = optarg;
        break;
      case 'F':  // fixed FPS = 20;
        fps_delay = 1000.f / (atoi(optarg) + 0.01);
        break;
      case 't':  // default 100
        MAX_TEST = atoi(optarg);
        break;
      case 'T':  // thread_type 0, 1, 2
        thread_type = atoi(optarg);
        break;
      case 'b':
        batch_sizes.push_back(atoll(optarg));
        break;
      case 'B':
        min_batch.push_back(atoll(optarg));
        break;        
      case 'v':  // video_frame_num 1, 10
        video_frame_num = atoi(optarg);
        break;
      case 'g':
        debug_output = true;  // print output
        break;
      case 'h':
        print_usage();
        return 0;
    }
  }

  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
  initLibNvInferPlugins(&trt_logger, "");

  void *pLibs = dlopen(pSoName, RTLD_LAZY);
  if (NULL == pLibs) {
    printf("Can not open library %s\n", pSoName);
    return -1;
  }

  {
    pCreateEngine = (PTCreateEngine)dlsym(pLibs, "CreateEngine");
    pRunEngine = (PTRunEngine)dlsym(pLibs, "RunEngine");
    pDestoryEngine = (PTDestoryEngine)dlsym(pLibs, "DestoryEngine");
    pGetBuffer = (PTGetBufferOfEngine)dlsym(pLibs, "GetBufferOfEngine");
  }

  if (NULL == pCreateEngine || NULL == pRunEngine || NULL == pDestoryEngine) {  
    printf("Can not load Function from %s\n", pSoName);
    dlclose(pLibs);
    return -1;
  }

  unsigned int engine_num = engine_filenames.size();
  char *output[16] = {NULL};
  int outsizes[16];
  IMGInfo input[16];
  memset(input, 0, sizeof(input));
  memset(outsizes, 0, sizeof(outsizes));
  int bufferNum[16];
  EngineBuffer *bufferInfo[16];

  // run models parallel
  for (unsigned int ch = 0; ch < engine_num; ch++) {  // ID for multi-engines
    const char *pMaskWeight = NULL;
    input[ch].modelType = 3;    
    if( ch < maskweight_filenames.size() ) {
      pMaskWeight = maskweight_filenames[ch].c_str();
      if (0 == strncasecmp(pMaskWeight, "Mask", 4) || 0 == strncasecmp(pMaskWeight, "ap_fsd", 6)) {
        input[ch].modelType = 1;
      } else if (strlen(pMaskWeight) <= 2) {
        input[ch].modelType = atoi(pMaskWeight);
      }
    }

    // default batchsize = 1
    if (batch_sizes.size() <= ch) batch_sizes.push_back(1);
    if (engine_chID.size() <= ch) engine_chID.push_back(-1);
    if (min_batch.size() <= ch) min_batch.push_back(batch_sizes[ch]);    

    if (input_filenames.size() > ch) {
      std::string input_filename = input_filenames[ch];
      int ret = 0;
      if (!input_filename.empty()) {
        if (0 == input_filename.compare(input_filename.size() - 4, 4, ".bin")
          || 0 == input_filename.compare(input_filename.size() - 4, 4, ".trt")) {
          auto t_start = std::chrono::high_resolution_clock::now();
          size_t size = ReadBinFile(input_filename.c_str(), input[ch].inputStream);
          float ms =
              std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();       
          printf("ReadBinFile to inputStream size=%lu Bytes cost=%.2f ms\n", size, ms); 
          
          delete[] input[ch].inputStream;         
          t_start = std::chrono::high_resolution_clock::now();
          size = ReadBinFileC(input_filename.c_str(), input[ch].inputStream);
          ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();       
          printf("ReadBinFileC to inputStream size=%lu Bytes cost=%.2f ms\n", size, ms);
          
          if (size <= 0) ret = -1;
        } else {  // Using Opencv to read image format
          ret = readImageBySTB(input_filename.c_str(), input[ch].rgbbuffer, input[ch].im_w, input[ch].im_h,
                                  input[ch].inputStream, input[ch].modelType);
        }

        if (0 != ret) {
          fprintf(stderr, "readImage Read image fail: %s\n", input_filename.c_str());
        }
      }
    }

    engine_chID[ch] = pCreateEngine(engine_chID[ch], engine_filenames[ch].c_str(), pMaskWeight);

    int &outputSize = outsizes[ch];
    if (NULL != pGetBuffer) {  // Get from GetBufferOfEngine.
      const char *sBufferType[ONNXTRT_MAX_BUFFERTYPE] = {"In:", "Out:", "Var:"};
      sBufferType[10]="CPU_In:";
      sBufferType[11]="CPU_Out:";      
    const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
    const char *sTensorFomat[6] = {"Linear", "CHW2", "HWC8", "CHW4", "CHW16", "CHW32"};
      pGetBuffer(engine_chID[ch], &bufferInfo[ch], &bufferNum[ch], NULL);
      printf("GetBuffer num = %d\n", bufferNum[ch]);
      for (int i = 0; i < bufferNum[ch]; i++) {
        EngineBuffer &bufInfo = bufferInfo[ch][i];
        const char *pName = bufInfo.name?bufInfo.name:"Unnamed";
        printf("Buf[%d]\t%s%-25.25s %dx[%d,%d,%d] %d, %s %s\n", i, sBufferType[bufInfo.nBufferType], pName, 
              batch_sizes[ch], bufInfo.d[0], bufInfo.d[1], bufInfo.d[2], bufInfo.nBufferSize, sDataType[bufInfo.nDataType], sTensorFomat[bufInfo.nTensorFormat]);
        if (0 < bufInfo.nBufferType) {                   // Output buffer
          if( input[ch].modelType >= 8 && input[ch].modelType <= 9) {
            outputSize += bufInfo.nBufferSize; // DLA has fixed batch, and output CHW16/CHW32
          } else { // GPU support batch <= batch_sizes[ch]
            int outSize = sizeof(float) * batch_sizes[ch];  // <= trt MaxBatch
            for (int j = 0; j < bufInfo.nDims; j++) outSize *= bufInfo.d[j];
            outputSize += std::min(bufInfo.nBufferSize, outSize);
          }
        } else { // input , check channel
          if ( 12 == bufInfo.d[0] && 2 * bufInfo.d[2] == input[ch].im_w ){ // png/jpg need res2channel if input ch = 12
            if( 4 == bufInfo.nTensorFormat ) {  // fp16-chw16 for DLA 
              transform_fp16_chw16( (float*)input[ch].inputStream, input[ch].im_w, input[ch].im_h );
            } else { // for normal GPU CHW
              res2channel( (float*)input[ch].inputStream, input[ch].im_w, input[ch].im_h );
            }
          }
          if( batch_sizes[ch] > 1 && input[ch].inputStream) {
            int im_size = 3*input[ch].im_w*input[ch].im_h;
            float *buffer = new float[im_size*batch_sizes[ch]];
            for (int nb = 0; nb < batch_sizes[ch]; nb++) {
              memcpy(buffer + nb * im_size, input[ch].inputStream, im_size * sizeof(float));
            }
            delete[] input[ch].inputStream;
            input[ch].inputStream = (char*)buffer;
          }
        }
      }
      printf("Tot outputSize=%d B\n", outputSize);
    } else if (0 == outputSize) {  // no outputSize set, get from environment
      char *val = getenv("TRT_OUTSIZE");
      if (NULL != val) {
        outputSize = atoi(val);
        printf("getenv TRT_OUTSIZE=%d\n", outputSize);
      }
    }

    if (MAX_TEST>10){
      pRunEngine(engine_chID[ch], batch_sizes[ch], input[ch].inputStream, 0, output[ch], input[ch].modelType);  // warm up
      printf("Warm up when MAX_TEST = %d > 10 , Init CudaGraph batch=%d.\n", MAX_TEST, batch_sizes[ch]);
      if( min_batch[ch] != batch_sizes[ch]){
        pRunEngine(engine_chID[ch], batch_sizes[ch], input[ch].inputStream, 0, output[ch], input[ch].modelType);  // warm up
        printf("Warm up when min_batch != batch_sizes. Init CudaGraph min_batch[%d]=%d.\n", ch, min_batch[ch]);
      }
    } 

    output[ch] = new char[outputSize];
    memset( output[ch], 0, outputSize );
  }

  if (MAX_TEST > 0) {
    auto t_begin = std::chrono::high_resolution_clock::now();   
    const float pct = 90.0f;
    std::vector<float> times(MAX_TEST);
    std::thread threads[16];
    if (0 == thread_type) {
      for (int nT = 0; nT < MAX_TEST; nT++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned int ch = 0; ch < engine_num; ch++) {
          int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
          threads[ch] = std::thread(testRunEngine, ch, engine_chID[ch], batch_size, input[ch].inputStream, 0, output[ch],
                                    input[ch].modelType);
        }
        for (unsigned int ch = 0; ch < engine_num; ch++) {
          threads[ch].join();
          if (debug_output) {
            checkOutput(ch, input[ch].modelType, output[ch], outsizes[ch]);
          }           
        }

        float ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
        printf("Tot time = %f\n", ms);
        times[nT] = ms;

        if (ms < fps_delay) {
          usleep((fps_delay - ms) * 1000);
        }
      }
    } else if (1 == thread_type || 11 == thread_type) {
      
      for (unsigned int ch = 0; ch < engine_num; ch++) {
        threads[ch] = std::thread(testRunEngine1, ch, engine_chID[ch], batch_sizes[ch], input[ch].inputStream, 0, video_frame_num,
                                  output[ch], input[ch].modelType);
        char thread_name[256];
        snprintf(thread_name, 255, "Test_Model%d", ch);
        pthread_setname_np(threads[ch].native_handle(), thread_name);
      }
      usleep(1000);
      for (int nT = 0; nT < MAX_TEST; nT++) {
        memset(g_times, 0, sizeof(g_times));
        float max_ms = 0.0f;
        g_start = std::chrono::high_resolution_clock::now();
        condvar.notify_all();
        for (unsigned int ch = 0; ch < engine_num; ch++) {
          for (int i = 0; i < 100; i++) {
            if (g_times[ch] < 0.1f)
              usleep(fps_delay * 1000 / 100);
            else if (11 == thread_type) {
              ch = engine_num;
              break;
            }

            max_ms = std::max(max_ms, g_times[ch]);

            if (debug_output) {
              checkOutput(ch, input[ch].modelType, output[ch], outsizes[ch]);
            }            
          }
        }
        float ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - g_start).count();
        printf("Max time = %f\n", max_ms);  // cout : 0.1ms, printf: 0.03ms
        printf("Tot time = %f\n", ms);
        times[nT] = max_ms;
        
        if (ms < fps_delay) {
          usleep((fps_delay - ms) * 1000);
        }        
      }
      ready = false;
      condvar.notify_all();
      for (unsigned int ch = 0; ch < engine_num; ch++) {
        threads[ch].join();
      }
    } else {  // no Thread, =2, sync output, =3 sync in/out or =-1 no sync
      for (int nT = 0; nT < MAX_TEST; nT++) {
        auto t_start = std::chrono::high_resolution_clock::now();   
        if (2 == thread_type || 3 == thread_type) {
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            auto inPtr = (2 == thread_type) ? input[ch].inputStream : nullptr;
            pRunEngine(engine_chID[ch], batch_size, inPtr, 0, nullptr, -1); // Feed data
          }
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            // pGetBuffer(ch, &pBufferInfo[ch], &bufNum[ch], NULL);
            pRunEngine(engine_chID[ch], batch_size, nullptr, -1, output[ch], input[ch].modelType); // Fetch output
            float ms =
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
            printf("CH.%d time = %f\n", ch, ms);
            if (debug_output) {
              checkOutput(ch, input[ch].modelType, output[ch], outsizes[ch]);
            }
          }
        } else if( -3 == thread_type ){
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            float tot_x = 0;
            float *outPtr = (float*)output[ch];
            for( int tst = 0; tst < 47; tst++)
            for( size_t idx = 0; idx < outsizes[ch] / sizeof(float); idx +=47 ){
              tot_x += sigmoid( outPtr[idx]);
            }
          }
        } else {  // -1: no sync, -2: no input/output
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            auto inPtr = (-1 == thread_type) ? input[ch].inputStream : nullptr;
            auto outPtr = (-1 == thread_type) ? output[ch] : nullptr;
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            pRunEngine(engine_chID[ch], batch_size, inPtr, 0, outPtr, 0);
            float ms =
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
            printf("[%d].chID[%d] time = %f\n", ch, engine_chID[ch], ms);
            if (debug_output) {
              checkOutput(ch, input[ch].modelType, outPtr, outsizes[ch]);
            }
          }
        }
        float ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
        printf("Tot time = %f\n", ms);
        times[nT] = ms;

        if (ms < fps_delay) {
          usleep((fps_delay - ms) * 1000);
        }
      }
    }
    float total_ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_begin).count();  
    percentile(pct, times, total_ms);
  }

  for (unsigned int ch = 0; ch < engine_num; ch++) {  // engine_num
    int modelType = input[ch].modelType;
    float *pOutData = (float *)output[ch];
    if (debug_output) {
      PrintOutBuffer(pOutData, modelType, batch_sizes[ch], bufferNum[ch], bufferInfo[ch]);
    }

    if (out_filenames.size() > ch) {
      std::string outfile = out_filenames[ch];

      for (int nb = 0; nb < batch_sizes[ch]; nb++) {  // process for every batch
        if (0 == out_filenames[ch].compare(out_filenames[ch].size() - 4, 4, ".ppm")) {
          std::ofstream file(out_filenames[ch].c_str(), std::ios::out | std::ios::binary);
          file << "P6\n" << input[ch].im_w << " " << input[ch].im_h << "\n255\n";
          file.write((char *)input[ch].rgbbuffer, (input[ch].im_w * input[ch].im_h) * 3);
          file.close();
        } else {  // Using Opencv to save other format
          saveImageByOpenCV(out_filenames[ch], input[ch].rgbbuffer, input[ch].im_w, input[ch].im_h);
        }
      }
    }
  }

  for (unsigned int ch = 0; ch < engine_num; ch++) {
    pDestoryEngine(engine_chID[ch]);
    if (NULL != input[ch].inputStream) {
      delete[] input[ch].inputStream;
    }
    if (NULL != output[ch]) {
      delete[] output[ch];
    }
  }

  dlclose(pLibs);
  
  RUNTIME_API_CALL(cudaDeviceSynchronize());
 
  // Flush CUPTI buffers before resetting the device.
  // This can also be called in the cudaDeviceReset callback. 
  finiTrace();

  return 0;
}