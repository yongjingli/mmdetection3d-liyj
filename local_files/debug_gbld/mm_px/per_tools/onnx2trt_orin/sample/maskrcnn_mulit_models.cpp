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
#include "cnpy.h"
#include "onnxtrt.h"
#include "NvInferPlugin.h"

PTCreateEngine pCreateEngineArray[ONNXTRT_MAX_ENGINENUM];
PTRunEngine pRunEngineArray[ONNXTRT_MAX_ENGINENUM];
PTDestoryEngine pDestoryEngineArray[ONNXTRT_MAX_ENGINENUM];
PTGetBufferOfEngine pGetBufferArray[ONNXTRT_MAX_ENGINENUM];

PTAllocateSpaceGPU pAllocateSpaceGPUArray[ONNXTRT_MAX_ENGINENUM];
PTMemcpyHost2DeviceGPU pMemcpyHost2DeviceGPUArray[ONNXTRT_MAX_ENGINENUM];

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

int getPreprocesParams(int &modelType, int &im_w, int &im_h, int *idxRGB, float *Scales, float *Means, int inType = 0) {
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

  if (3 <= modelType && inType == 0) {
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
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int readImageBySTB(const char *filename, unsigned char *&rgbbuffer, int &im_w, int &im_h, int &im_ch, char *&databuffer,
                    int inputType = 0,  int modelType = 1) {

  if(inputType == 1 || inputType == 4) {
    fprintf(stderr, "read Image does not support for input tpye 1 and 4. Return.\n");
    return -1;
  }
  im_ch = 3;
  rgbbuffer = stbi_load(filename, &im_w, &im_h, &im_ch, 0);
  int real_ch = im_ch;
  int ch = im_ch == 4 ? 3 : im_ch;
  im_ch  = im_ch == 4 ? 3 : im_ch;
  // bool fourChInput = im_ch == 4 ? true : false;
  int imgsize = im_w * im_h;
  int real_size = imgsize * real_ch;
  int size = imgsize * ch;    
  printf("readImage %s by stbLib, size = %d (%dx%dx%d)\n", filename, real_size, real_ch, im_h, im_w);
  if( 0 == im_w ){
    fprintf(stderr, "readImageBySTB read image fail: %s\n", filename);
    return -1;
  }      

  int   idxRGB[3] = {0, 1, 2};

  if(inputType == 3 || inputType == 5) {
    // databuffer = reinterpret_cast<char*>(rgbbuffer);
    databuffer = new char[size];

    for (int i = 0; i < imgsize; ++i) 
      for (int j = 0; j < ch; ++j)
        databuffer[i + j * imgsize] =(reinterpret_cast<char*>(rgbbuffer)[i * real_ch + idxRGB[j]]);

    return 0;
  }

  if (inputType == 2) {
    databuffer = new char[size * sizeof(float)];

    for (int i = 0; i < imgsize; ++i) 
      for(int j = 0; j < ch; ++j) 
        reinterpret_cast<float*>(databuffer)[i + j * imgsize] = 1.f * (reinterpret_cast<char*>(rgbbuffer)[i * real_ch + idxRGB[j]]);
    return 0;
  }

  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  getPreprocesParams(modelType, im_w, im_h, idxRGB, Scales, Means, inputType);


  float *floatbuffer = new float[size];


  for (int i = 0; i < imgsize; i++) {
    for(int j = 0; j < ch; ++j) {
      floatbuffer[i + imgsize * j] = (rgbbuffer[i * real_ch + idxRGB[j]] - Means[j]) / Scales[j]; 
    }
  }
  
  
  databuffer = (char *)floatbuffer;

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
  if (NULL == pRunEngineArray[ch]) return -1;

  auto t_start = std::chrono::high_resolution_clock::now();

  int nRet = pRunEngineArray[ch](engine_chID, batch, inputData, inputType, outData, outType);

  float ms = std::chrono::duration<float, std::milli>(  // 0.4us
                 std::chrono::high_resolution_clock::now() - t_start)
                 .count();
  printf("CH[%d] time = %f\n", ch, ms);  // cout : 0.1ms, printf: 0.03ms

  return nRet;
}

void checkShort(short *outRef, short* outData, int size, int ulp=4){
      
  int count_notEq = 0;
  int eltCount = size / sizeof(short);
  for( int i = 0; i< eltCount; i++){
    if(std::abs(outData[i] - outRef[i]) >= ulp){
      printf("%d Not Equal: %d vs %d\n", i, outData[i], outRef[i]);
      if( ++count_notEq >= 16 ) break;
    }
  }
  printf("checkOutput count_notEq = %d\n", count_notEq );
}

// Check output if unstable
void checkOutput(int ch, int outType, char* outptr, int size){
printf("Checking output.\n");
static char* outRefs[16] = {nullptr};

  if( NULL == outptr ){
    return;
  } else if ( 0 == size && outRefs[ch] ) { // last run, free
    delete[] outRefs[ch];
    outRefs[ch] = NULL;
    return;
  } else if( NULL == outRefs[ch] ){ // first run, alloc and copy
    outRefs[ch] = new char[size];
    memcpy( outRefs[ch], outptr, size );
    return;
  } 
   printf("checkOutput outType = %d\n", outType ); 
  if( 9 == outType ){
    checkShort((short*)outRefs[ch], (short*)outptr, size );
  }

printf("Checking output finish.\n");
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
    nRet = pRunEngineArray[ch](engine_chID, batch, inputData, inT, outData, outType);

    g_times[ch] = std::chrono::duration<float, std::milli>(  // 0.4us
                      std::chrono::high_resolution_clock::now() - g_start)
                      .count();
    printf("CH[%d] time = %f\n", ch,
           g_times[ch]);  // cout : 0.1ms, printf: 0.03ms
    max_ms = g_times[ch];
    {
      std::unique_lock<std::mutex> lck(mtx);
      condvar.wait(lck);
    }
  };

  return nRet;
}

#include "half_convert.h"
static half2float g_h2f;
static float2half g_f2h;

void index2Sub(int idx, std::vector<int>& sub, std::vector<int> newDims) {
  for(int i = newDims.size() - 2; i >= 0; i--) {
        sub[i] = idx % newDims[i];
        idx    = idx - sub[i];
        idx    = idx / newDims[i];
    }

}

// print the fist & last 64(PrintNum) data
// print the fist & last 64(PrintNum) data
template <typename T>
void printOutput(int64_t eltCount, T *outputs, EngineBuffer bufferInfo, bool trans, float scale, int bits = 32) {
  const int PrintNum = 64;
  printf("the elt count is: %ld\n", eltCount);
  // printf("the elt count is: ");
  // for(int i = 0; i < bufferInfo.nDims; ++i)
  //   printf("%d,\t", bufferInfo.d[i]);
  // printf("\n");
  for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {

    int newEltIdx;
    // int chwIdx;
    if(!trans) {
      newEltIdx = eltIdx;
    } else if(trans && bits == 8) {
      std::vector<int> subScript(bufferInfo.nDims, 0);
      std::vector<int> newDims  (bufferInfo.nDims, 0);
      newDims[2] = bufferInfo.d[2];
      newDims[1] = bufferInfo.d[1];
      newDims[0] = bufferInfo.d[0];
      index2Sub(eltIdx, subScript, newDims);
      // printf("%d\t",newDims.size());
      // printf("%d-%d-%d\t", bufferInfo.d[0], bufferInfo.d[1], bufferInfo.d[2]);
      // printf("%d-%d-%d\t", subScript[0], subScript[1], subScript[2]);
      int subChannel   = subScript[0] % 32;
      int groupChannel = subScript[0] / 32;
      newEltIdx = groupChannel * 32 * bufferInfo.d[2] * bufferInfo.d[1] + 
                  subScript[1] * 32 * bufferInfo.d[2] + 
                  subScript[2] * 32 + 
                  subChannel;
      // chwIdx = subScript[0] * bufferInfo.d[2] * bufferInfo.d[1] + subScript[1] * bufferInfo.d[2] + subScript[2];
    } else if(trans && bits == 16) {
      std::vector<int> subScript(bufferInfo.nDims, 0);
      std::vector<int> newDims  (bufferInfo.nDims, 0);
      newDims[2] = bufferInfo.d[2];
      newDims[1] = bufferInfo.d[1];
      newDims[0] = bufferInfo.d[0];
      index2Sub(eltIdx, subScript, newDims);
      // printf("%d-%d-%d\t", subScript[0], subScript[1], subScript[2]);
      int subChannel   = subScript[0] % 16;
      int groupChannel = subScript[0] / 16;
      newEltIdx = groupChannel * 16 * bufferInfo.d[2] * bufferInfo.d[1] + 
                  subScript[1] * 16 * bufferInfo.d[2] + 
                  subScript[2] * 16 + 
                  subChannel;
    }

    // printf("%d-%d\t",eltIdx, newEltIdx);

    if (8 == bits) {
      if (scale < 0.0f)
        std::cerr << std::hex << (int)outputs[newEltIdx] << "\t, ";
      else {
        float result = reinterpret_cast<signed char*>(outputs)[newEltIdx] * scale;
        printf("%.2lf\t, ", result);
      }
      
    }
      
    else if (16 == bits)
      std::cerr << h2f_internal(outputs[newEltIdx]) << "\t, ";
    else
      std::cerr << outputs[newEltIdx] << "\t, ";
  }
  if (eltCount > PrintNum) {
    // if (eltCount < 0) {
      std::cerr << " ... ";
      for (int64_t eltIdx = (eltCount - PrintNum); eltIdx < eltCount; ++eltIdx) {

        int newEltIdx;
        if(!trans) {
          newEltIdx = eltIdx;
        } else if(trans && bits == 8) {
          std::vector<int> subScript(bufferInfo.nDims, 0);
          std::vector<int> newDims  (bufferInfo.nDims, 0);
          newDims[2] = bufferInfo.d[2];
          newDims[1] = bufferInfo.d[1];
          newDims[0] = bufferInfo.d[0];
          index2Sub(eltIdx, subScript, newDims);
          // printf("%d\t",newDims.size());
          // printf("%d-%d-%d\t", bufferInfo.d[0], bufferInfo.d[1], bufferInfo.d[2]);
          // printf("%d-%d-%d\t", subScript[0], subScript[1], subScript[2]);
          int subChannel   = subScript[0] % 32;
          int groupChannel = subScript[0] / 32;
          newEltIdx = groupChannel * 32 * bufferInfo.d[2] * bufferInfo.d[1] + 
                      subScript[1] * 32 * bufferInfo.d[2] + 
                      subScript[2] * 32 + 
                      subChannel;
          // chwIdx = subScript[0] * bufferInfo.d[2] * bufferInfo.d[1] + subScript[1] * bufferInfo.d[2] + subScript[2];
        } else if(trans && bits == 16) {
          std::vector<int> subScript(bufferInfo.nDims, 0);
          std::vector<int> newDims  (bufferInfo.nDims, 0);
          newDims[2] = bufferInfo.d[2];
          newDims[1] = bufferInfo.d[1];
          newDims[0] = bufferInfo.d[0];
          index2Sub(eltIdx, subScript, newDims);
          // printf("%d-%d-%d\t", subScript[0], subScript[1], subScript[2]);
          int subChannel   = subScript[0] % 16;
          int groupChannel = subScript[0] / 16;
          newEltIdx = groupChannel * 16 * bufferInfo.d[2] * bufferInfo.d[1] + 
                      subScript[1] * 16 * bufferInfo.d[2] + 
                      subScript[2] * 16 + 
                      subChannel;
        }

        // printf("%d-%d\t",eltIdx, newEltIdx);

        if (8 == bits) {
          if (scale < 0.0f)
            std::cerr << std::hex << (int)outputs[newEltIdx] << "\t, ";
          else {
            float result = reinterpret_cast<signed char*>(outputs)[newEltIdx] * scale;
            printf("%.2lf\t, ", result);
          }
        }
          
        else if (16 == bits)
          std::cerr << h2f_internal(outputs[newEltIdx]) << "\t, ";
        else
          std::cerr << outputs[newEltIdx] << "\t, ";
      }
  }

  std::cerr << std::endl;
}
//Print the value of filter
void printFilter(int nDims, int d[], float *outputs) {
  const int PrintNum = 16;
  int eltCount = d[1];
  for( int i = 0; i < d[0]; i ++) {
    int index = ((int*)outputs)[0];
    if( index < 0 ) {
      break;
    }
    std::cerr << i << ": Idx."<< index << "\t: ";
    for (int eltIdx = 1; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {
        std::cerr << outputs[eltIdx] << "\t, ";
    }
    if (eltCount > PrintNum) {
      if (eltCount > 2 * PrintNum) {
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
void PrintOutBuffer(float *pBuf, int outType, int batchsize, int bufferNum, EngineBuffer *bufferInfo,
                    bool transpose_output, float scale) {
  int valNum = 0;
  if (2 == outType) {  // packed buffers by batchsize
    for (int nb = 0; nb < batchsize; nb++) {
      for (int i = 0; i < bufferNum; i++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++) eltCount *= bufferInfo[i].d[d];
        printf("[%d]%s.%d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
        printOutput(eltCount, pBuf + valNum, bufferInfo[i], transpose_output, scale);
        valNum += eltCount;
      }
    }
  } else if (8 == outType || 9 == outType) {  // for int8 or fp16

     for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        // int eltCount = bufferInfo[i].nBufferSize / 2;
        int eltCount = 0;

        if (9 == outType) {
          int totChannel = ((bufferInfo[i].d[0] - 1 ) / 16 + 1) * 16;
          eltCount = totChannel * bufferInfo[i].d[1] * bufferInfo[i].d[2];
          valNum = nb * eltCount * 2;
          int realElt = bufferInfo[i].d[0] * bufferInfo[i].d[1] * bufferInfo[i].d[2];
          printf("[%d]%s.%d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
          printOutput(realElt, (unsigned short *)(bufferInfo[i].p + valNum), bufferInfo[i], transpose_output, scale, 16);
        }
        else {
          int totChannel = ((bufferInfo[i].d[0] - 1) / 32 + 1) * 32;
          eltCount = totChannel * bufferInfo[i].d[1] * bufferInfo[i].d[2];
          valNum = nb * eltCount * 4; 
          int realElt = bufferInfo[i].d[0] * bufferInfo[i].d[1] * bufferInfo[i].d[2];
          printf("[%d]%s.%d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
          printOutput(realElt, (unsigned short *)(bufferInfo[i].p + valNum), bufferInfo[i], transpose_output, scale, 8);
        }
          
      }
    }
  } else  {  // plan buffers, default
    for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++) eltCount *= bufferInfo[i].d[d];
        printf("[%d]%s.batchId: %d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
        if( strstr(bufferInfo[i].name, "_filter")!=NULL){
          printFilter(bufferInfo[i].nDims, bufferInfo[i].d, pBuf + valNum);
        } else {
          printOutput(eltCount, pBuf + valNum, bufferInfo[i], transpose_output, scale);
        }
        valNum += eltCount;
      }
    }
  }
}

// convert 3 channel to 12 channel
int res2channel( float *pSrc, int im_w, int im_h, int im_ch ,int input_type ) {
  if( nullptr == pSrc || 0 == im_w || 0 == im_h ){
    return -1;
  }  

  float scale = input_type == 2 ? 255.f : 1.f;

  printf("res2channel : %dx%dx%d -> %dx%dx%d\n", im_ch, im_h, im_w, im_ch * 4, im_h / 2, im_w / 2);
  int im_size = im_ch * im_w * im_h;
  float *pDst = new float[im_size];
  int src_idx=0;
  for( int c = 0; c < im_ch; c++){
    for( int y = 0; y < im_h; y++) {
      for( int x = 0; x< im_w; x++) {
        int dst_c = ((x & 1) * 2 + (y & 1)) * im_ch + c;
        int dst_y = y / 2;
        int dst_x = x / 2;
        int dst_idx = dst_c * (im_w / 2) * (im_h / 2) + dst_y * (im_w / 2) + dst_x;    

        pDst[dst_idx] = pSrc[src_idx ++] / scale;
      }
    }
  }

  // for(int i = 0; i < 64; ++i)
  //   printf("%.6lf\t, ", pDst[i]);
  memcpy( pSrc, pDst, im_size * sizeof(float) );
  delete[] pDst;

  return 0;
}

// convert 2channel float-chw to 12channel fp16-chw16 or int8-chw32
int transform_DLA( float *pSrc, int im_w, int im_h, int im_ch, int type) {
  if( nullptr == pSrc || 0 == im_w || 0 == im_h ){
    return -1;
  }  

  int CALIG = (1 == type) ? 16 : 32;
  int elt_size = (1 == type) ? 2 : 1;  
  int in_size = im_ch * im_w * im_h * sizeof(float); // fp32-chw
  int out_size = CALIG * (im_w / 2) * (im_h / 2) * elt_size; // fp16-chw16 or int8-chw32
  const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
  printf("transform_DLA %dx%d in:%d B, out:%d B, type:%s\n", im_w, im_h, in_size, out_size, sDataType[type]);

  char *pDst = new char[out_size];
  memset( pDst, 0, out_size);
  int src_idx = 0;
  for( int c = 0; c < im_ch; c++){
    for( int y = 0; y < im_h; y++) {
      for( int x = 0; x< im_w; x++) {
        int dst_c = ((x & 1) * 2 + (y & 1)) * im_ch + c;
        int dst_y = y / 2;
        int dst_x = x / 2;
        int dst_idx = (dst_y * (im_w / 2) + dst_x)*CALIG + dst_c;    
        if(1 == type) {
          ((unsigned short*)pDst)[dst_idx] = g_f2h.convert(pSrc[src_idx ++]);
        } else {
          pDst[dst_idx] = pSrc[src_idx ++] * 127;
        }
      }
    }
  }
  memset( pSrc, 0, in_size);  
  memcpy( pSrc, pDst, out_size );
  delete[] pDst;

  return 0;  
}


inline float sigmoid(float x) { return (1 / (1 + expf(-x))); }

typedef struct IMGInfo {
  int im_w = 0;
  int im_h = 0;
  int im_ch = 0;
  unsigned char *rgbbuffer = NULL;
  char *inputStream;
  int modelType;  // 0: LLD, 1: MaskRCNN/FasterRCNN, 8: Byte output
  int inputType;
} IMGInfo;

int main(int argc, char *argv[]) {
  if (argc <= 1) print_usage();
  std::vector<std::string> engine_filenames;
  std::vector<std::string> maskweight_filenames;
  std::vector<std::string> input_filenames;
  std::vector<std::string> out_filenames;
  std::vector<int> batch_sizes, min_batch;
  std::vector<int> engine_chID;
  std::vector<LidarInput> lidarInput;

  float fps_delay = 10;  // delay ms to get fixed FPS
  // const char *pSoNameArray[ONNXTRT_MAX_ENGINENUM] = {"./libonnxtrt_std.so"};
  std::vector<const char *> pSoNameArray;
  bool debug_output = false;
  int MAX_TEST = 100;
  int thread_type = 2; // // Default thread type = 2, mulit-stream in one thread
  int video_frame_num = 1;  // 1:single image, >1: frame number of video
  std::vector<int> inputTypeArray;
  float scale = -999.99f; 
  bool transpose_output = false; 
  bool doRes2Ch = true;

  std::vector<bool> test_tsl(ONNXTRT_MAX_ENGINENUM, false);

  int arg = 0;
  while ((arg = ::getopt(argc, argv, "o:e:c:i:m:s:b:B:F:t:T:v:G:S:l:pghj")) != -1) {
    if ('o' == arg || 'e' == arg || 'i' == arg || 'm' == arg || 's' == arg || 'b' == arg || 'F' == arg || 't' == arg ||
        'T' == arg || 'c' == arg || 'v' == arg || 'B' == arg || 'G' == arg || 'S' == arg || 'l' == arg) {
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
        pSoNameArray.push_back(optarg);
        break;
      case 'S':
        scale = atof(optarg);
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
      case 'G': //input type of model
        inputTypeArray.push_back(atoi(optarg));
        break;
      case 'g':
        debug_output = true;  // print output
        break;
      case 'p':
        transpose_output = true;
        break;
      case 'j':
        doRes2Ch = false;
        break;
      case 'l':
        test_tsl[atoi(optarg) - 1] = true;
        break;
      case 'h':
        print_usage();
        return 0;
    }
  }

  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  TRT_Logger trt_logger((nvinfer1::ILogger::Severity)verbosity);
  initLibNvInferPlugins(&trt_logger, "");

  if(engine_filenames.size() != 1) {
    if(engine_filenames.size() != pSoNameArray.size()) {
      std::cout<<"Error:The lib so num mismatches with engine num. Return."<<std::endl;
      return -1;
    } else if(engine_filenames.size() != batch_sizes.size()) {
      std::cout<<"Error:The batch size num mismatches with engine num. Return."<<std::endl;
      std::cout<<"Note: You have to declare batch size explicitly for each engines."<<std::endl;
      return -1;
    }else if(engine_filenames.size() != inputTypeArray.size()) {
      std::cout<<"Error:The input type num mismatches with engine num. Return."<<std::endl;
      std::cout<<"Note: You have to declare input type explicitly for each engines."<<std::endl;
      return -1;
    }
  } else {
    if(inputTypeArray.size() == 0) {
      inputTypeArray.push_back(0);
    }
  }

  size_t totBatchNum = 0; 

  for(size_t i = 0; i < batch_sizes.size(); ++i)
    totBatchNum += batch_sizes[i];

  std::cout<<"Tot batch number is "<<totBatchNum<<std::endl;

  if(totBatchNum != input_filenames.size() && engine_filenames.size() != 1) {
    std::cout<<"Error:The batch size num mismatches with inputfile num. Return."<<std::endl;
    return -1;
  }

  std::vector<void *> pLibsArray(ONNXTRT_MAX_ENGINENUM);
  for(size_t i = 0; i < pSoNameArray.size(); ++i){
    pLibsArray[i] = dlopen(pSoNameArray[i], RTLD_LAZY);
    if (NULL == pLibsArray[i]) {
      printf("Can not open %zu th library %s\n", i, pSoNameArray[i]);
      return -1;
    }

    pCreateEngineArray[i]         = (PTCreateEngine)dlsym(pLibsArray[i], "CreateEngine");
    pRunEngineArray[i]            = (PTRunEngine)dlsym(pLibsArray[i], "RunEngine");
    pDestoryEngineArray[i]        = (PTDestoryEngine)dlsym(pLibsArray[i], "DestoryEngine");
    pGetBufferArray[i]            = (PTGetBufferOfEngine)dlsym(pLibsArray[i], "GetBufferOfEngine");

    pAllocateSpaceGPUArray[i]     = (PTAllocateSpaceGPU)dlsym(pLibsArray[i], "AllocateSpaceGPU");
    pMemcpyHost2DeviceGPUArray[i] = (PTMemcpyHost2DeviceGPU)dlsym(pLibsArray[i], "MemcpyHost2DeviceGPU");

    if (NULL == pCreateEngineArray[i] || NULL == pCreateEngineArray[i] || NULL == pDestoryEngineArray[i]) {  
      printf("Can not load Function from %zuth %s\n", i, pSoNameArray[i]);
      dlclose(pLibsArray[i]);
      return -1;
    }

    if(NULL == pAllocateSpaceGPUArray[i] || NULL == pMemcpyHost2DeviceGPUArray[i]) {
      printf("Warning: Can not load GPU memory function from %zu th %s\n", i, pSoNameArray[i]);
      
      if(inputTypeArray[i] == 2 || inputTypeArray[i] == 3) {
        printf("Error: Since the GPU memory functions cannot be load, so the input type cannot be specified as 2 or 3. Return.");
        dlclose(pLibsArray[i]);
        return -1;
      }

    }
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

    PTCreateEngine pCreateEngine = pCreateEngineArray[ch];
    PTGetBufferOfEngine pGetBuffer = pGetBufferArray[ch];
    PTRunEngine pRunEngine = pRunEngineArray[ch];


    PTAllocateSpaceGPU pAllocateSpaceGPU = pAllocateSpaceGPUArray[ch];
    PTMemcpyHost2DeviceGPU pMemcpyHost2DeviceGPU = pMemcpyHost2DeviceGPUArray[ch];

    int inputType = inputTypeArray[ch];


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
  


    int tot_privous_batch = 1;
    for(unsigned int iter = 0; iter < ch; ++iter)
      tot_privous_batch *= batch_sizes[iter];

    tot_privous_batch = ch == 0 ?  0 : tot_privous_batch;

    size_t cur_batch_acc = ch == 0 ? batch_sizes[ch] : tot_privous_batch + batch_sizes[ch];

    if(!doRes2Ch)
      printf("Warning: Skiping Res2Channel.\n");
  
    if (input_filenames.size() > ch) {
      std::string input_filename = input_filenames[ch];
      int ret = 0;
      if (!input_filename.empty()) {
        if (0 == input_filename.compare(input_filename.size() - 4, 4, ".bin")
           || 0 == input_filename.compare(input_filename.size() - 4, 4, ".trt")) {    

          auto t_start = std::chrono::high_resolution_clock::now();
          size_t size = ReadBinFileC(input_filename.c_str(), input[ch].inputStream);
          float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();       
          printf("ReadBinFileC to inputStream size=%lu Bytes cost=%.2f ms\n", size, ms);
          
          if(inputType == 4) {
            LidarInput lInputTmp;
            lInputTmp.pointCloud = (void*) input[ch].inputStream;
            lInputTmp.pointNum   = size / 16;
            lidarInput.push_back(lInputTmp);
            input[ch].inputStream = reinterpret_cast<char*>(&lidarInput[ch]);
          }

          if (size <= 0) ret = -1;
        } else {  // Using Opencv to read image format
         if(batch_sizes[ch] == 1)
          ret = readImageBySTB(input_filename.c_str(), input[ch].rgbbuffer, input[ch].im_w, input[ch].im_h,
                                input[ch].im_ch,  input[ch].inputStream, inputType, input[ch].modelType);
          else if(cur_batch_acc <= input_filenames.size()){
            printf("Read data of multiple batches.\n");
            int imageWidth, imageHeight, imageChannel;
            unsigned char* rgbBuffer   = NULL;
            char*          inputStream = NULL;


            int copyOffset = 0;
            int copyByte   = (inputType == 5 || inputType == 3) ? sizeof(char) : sizeof(float);
            int imageSize  = 0;

            for(int iter = 0; iter < batch_sizes[ch]; ++iter) {
              int imagePos = tot_privous_batch + iter;
              ret = readImageBySTB(input_filenames[imagePos].c_str(), rgbBuffer, 
              imageWidth, imageHeight, imageChannel, inputStream, inputType, input[ch].modelType);

              if(iter == 0) {
                input[ch].rgbbuffer = new unsigned char[imageWidth * imageHeight * imageChannel * batch_sizes[ch]];

                input[ch].inputStream = new char[imageWidth * imageHeight * imageChannel  * batch_sizes[ch] * copyByte];

                imageSize = imageHeight * imageWidth * imageChannel;

                input[ch].im_ch = imageChannel;
                input[ch].im_w  = imageWidth;
                input[ch].im_h  = imageHeight;
              }

              memcpy(input[ch].rgbbuffer   + copyOffset, rgbBuffer, imageSize);
              memcpy(input[ch].inputStream + copyOffset * copyByte, inputStream, imageSize * copyByte);

              copyOffset += imageSize;

              delete[] rgbBuffer;
              delete[] inputStream;

              rgbBuffer  = NULL;
              inputStream = NULL;

            }

          } else {
            ret = readImageBySTB(input_filename.c_str(), input[ch].rgbbuffer, input[ch].im_w, input[ch].im_h,
                                input[ch].im_ch,  input[ch].inputStream, inputType, input[ch].modelType);
            printf("Warning: batch size and input file num does not match. Will fill buffer using first image.\n");
          }
        }

        if (0 != ret) {
          fprintf(stderr, "readImage Read image fail: %s\n", input_filename.c_str());
        }
      }
    }
    // fprintf(stderr, "Creating Engine\n");
    engine_chID[ch] = pCreateEngine(engine_chID[ch], engine_filenames[ch].c_str(), pMaskWeight);
    // fprintf(stderr, "Engine Created\n");

    int &outputSize = outsizes[ch];
    if (NULL != pGetBuffer) {  // Get from GetBufferOfEngine.
    printf("Geting Buffer\n");
    const char *sBufferType[ONNXTRT_MAX_BUFFERTYPE] = {"In:", "Out:", "Var:"};
	  sBufferType[10]="CPU_In:";
	  sBufferType[11]="CPU_Out:";      
    const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
    const char *sTensorFomat[6] = {"Linear", "CHW2", "HWC8", "CHW4", "CHW16", "CHW32"};
      pGetBuffer(engine_chID[ch], &bufferInfo[ch], &bufferNum[ch], NULL);
      printf("GetBuffer num = %d\n", bufferNum[ch]);
      for (int i = 0; i < bufferNum[ch]; i++) {
        EngineBuffer &bufInfo = bufferInfo[ch][i];
        const char *pName = bufInfo.name ? bufInfo.name : "Unnamed";
        printf("Buf[%d]\t%s%-25.25s %dx[%d,%d,%d] %d, %s %s\n", i, sBufferType[bufInfo.nBufferType], pName, 
               batch_sizes[ch], bufInfo.d[0], bufInfo.d[1], bufInfo.d[2], bufInfo.nBufferSize, sDataType[bufInfo.nDataType], sTensorFomat[bufInfo.nTensorFormat]);
        if (0 < bufInfo.nBufferType) {                   // Output buffer
          if( input[ch].modelType >= 8 && input[ch].modelType <= 9) {
            outputSize += (bufInfo.nBufferSize * batch_sizes[ch]); // DLA has fixed batch, and output CHW16/CHW32
          } else { // GPU support batch <= batch_sizes[ch]
            int outSize = sizeof(float) * batch_sizes[ch];  // <= trt MaxBatch
            for (int j = 0; j < bufInfo.nDims; j++) outSize *= bufInfo.d[j];
              outputSize += std::max(bufInfo.nBufferSize, outSize);
          }
        } else { // input , check channel
          
          if(inputType != 3) {
            if ( bufInfo.d[0] / 4 == input[ch].im_ch && 2 * bufInfo.d[2] == input[ch].im_w && inputType != 4){ // png/jpg need res2channel if input ch = 12
              if( 1 <= bufInfo.nDataType) {  // For DLA, 1:fp16-chw16, 2: int8-chw32  
                transform_DLA( (float*)input[ch].inputStream, input[ch].im_w, input[ch].im_h, input[ch].im_ch, bufInfo.nDataType );
              } else { // for normal GPU CHW
                  if(batch_sizes[ch] > 1 && cur_batch_acc <= input_filenames.size()) {
                    int res2cOffset = 0;
                    for(int ri = 0; ri < batch_sizes[ch]; ++ri) {
                      res2channel( ((float*)input[ch].inputStream) + res2cOffset, input[ch].im_w, input[ch].im_h, input[ch].im_ch, inputType);
                      res2cOffset += input[ch].im_w * input[ch].im_h * input[ch].im_ch;
                    } 
                  } else {
                    res2channel( (float*)input[ch].inputStream, input[ch].im_w, input[ch].im_h, input[ch].im_ch, inputType);
                  }
              }
            }
          } 

          if( batch_sizes[ch] > 1 && input[ch].inputStream && cur_batch_acc > input_filenames.size()) {
            printf("Filling buffer using one image...\n");
            int im_size = input[ch].im_ch * input[ch].im_w * input[ch].im_h;
            int bytes   = (inputType == 5 || inputType == 3) ? sizeof(char) : sizeof(float);
            void *buffer = new char[im_size * batch_sizes[ch] * bytes];
            for (int nb = 0; nb < batch_sizes[ch]; nb++) {
              memcpy(buffer + nb * im_size, input[ch].inputStream, im_size * bytes);
            }
            delete[] input[ch].inputStream;
            input[ch].inputStream = (char*)buffer;
          }
          
          if(inputType == 2 || inputType == 3) {
              int eltSize = (inputType == 2) ? sizeof(float) : sizeof(char);  
              int copySize = batch_sizes[ch] * input[ch].im_ch * input[ch].im_w * input[ch].im_h;
              void* gpuPtr = pAllocateSpaceGPU(eltSize, copySize);
              pMemcpyHost2DeviceGPU(gpuPtr, input[ch].inputStream, eltSize, copySize);
              delete[] input[ch].inputStream;
              input[ch].inputStream = (char*) gpuPtr;
              printf("Copied input data to the GPU.\n");
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

    // printf("Warming up\n");
    if (MAX_TEST>10){
      pRunEngine(engine_chID[ch], batch_sizes[ch], input[ch].inputStream, inputType, output[ch], input[ch].modelType);  // warm up
      printf("Warm up when MAX_TEST = %d > 10 , Init CudaGraph batch=%d.\n", MAX_TEST, batch_sizes[ch]);
      if( min_batch[ch] != batch_sizes[ch]){
        pRunEngine(engine_chID[ch], batch_sizes[ch], input[ch].inputStream, inputType, output[ch], input[ch].modelType);  // warm up
        printf("Warm up when min_batch != batch_sizes. Init CudaGraph min_batch[%d]=%d.\n", ch, min_batch[ch]);
      }
    } 
    // printf("Warmed up\n");
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
          int inputType = test_tsl[ch] && (nT !=0) ? (65536 | inputTypeArray[ch]) : inputTypeArray[ch];
          int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
          threads[ch] = std::thread(testRunEngine, ch, engine_chID[ch], batch_size, input[ch].inputStream, inputType, output[ch],
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
        int inputType = inputTypeArray[ch];
        threads[ch] = std::thread(testRunEngine1, ch, engine_chID[ch], batch_sizes[ch], input[ch].inputStream, inputType, video_frame_num,
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
          int inputType = test_tsl[ch] && (nT !=0) ? (65536 | inputTypeArray[ch]) : inputTypeArray[ch];
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

            int inputType = test_tsl[ch] && (nT !=0) ? (65536 | inputTypeArray[ch]) : inputTypeArray[ch];
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            auto inPtr = (2 == thread_type) ? input[ch].inputStream : nullptr;
            pRunEngineArray[ch](engine_chID[ch], batch_size, inPtr, inputType, nullptr, -1); //feed data
          }
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            // pGetBuffer(ch, &pBufferInfo[ch], &bufNum[ch], NULL);
            pRunEngineArray[ch](engine_chID[ch], batch_size, nullptr, -1, output[ch], input[ch].modelType); // Fetch output
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
            int inputType = test_tsl[ch] && (nT !=0) ? (65536 | inputTypeArray[ch]) : inputTypeArray[ch];
            auto inPtr = (-1 == thread_type) ? input[ch].inputStream : nullptr;
            auto outPtr = (-1 == thread_type) ? output[ch] : nullptr;
            int batch_size = (nT%2) ? batch_sizes[ch] : min_batch[ch]; // for main/narrow 20Hz/10Hz 
            pRunEngineArray[ch](engine_chID[ch], batch_size, inPtr, inputType, outPtr, 0);
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
      PrintOutBuffer(pOutData, modelType, batch_sizes[ch], bufferNum[ch], bufferInfo[ch], transpose_output, scale);
    }

    if (out_filenames.size() > ch) {
      std::string outfile = out_filenames[ch];

      for (int nb = 0; nb < batch_sizes[ch]; nb++) {  // process for every batch
#if 0                                                 // Draw result, need update             
        unsigned char *rgbbuffer = input[ch].rgbbuffer;
        int im_w = input[ch].im_w;
        int im_h = input[ch].im_h;
        if (1 == modelType) {
          std::vector<TRoiMaskOut> roiMasks;
          ParseMaskData(pOutData + nb * MaskBlockSize, roiMasks);
          if (NULL != rgbbuffer) {
            DrawMaskResult(roiMasks, rgbbuffer, im_w, im_h, outfile.c_str());
          }
        } else {
          DrawLLDResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str(),
                        outsizes[ch]);
        }
#endif
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
    int inputType = inputTypeArray[ch];
    pDestoryEngineArray[ch](engine_chID[ch]);
    if (NULL != input[ch].inputStream && inputType == 0) {
      delete[] input[ch].inputStream;
    } else if(NULL != input[ch].inputStream && inputType == 4) {
      if (NULL != lidarInput[ch].pointCloud)
            delete[] lidarInput[ch].pointCloud;
    }
    if (NULL != output[ch]) {
      delete[] output[ch];
    }

    dlclose(pLibsArray[ch]);
  }

  

  return 0;
}
