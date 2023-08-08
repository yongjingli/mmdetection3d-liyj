
#include "helper.h"
#include "inference_client.h"

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <thread>
#include <dlfcn.h>

PTCreateEngine pCreateEngine;
PTRunEngine pRunEngine;
PTDestoryEngine pDestoryEngine;
PTGetBufferOfEngine pGetBuffer;
PTClientQueryCallback pClientQueryCallback;


#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void spatial2Channel(unsigned char* in, float* out, int width, int height, int channel);
void clientThreadQueryCallback(vector<int> engineId);
void printOutBuf(float *pBuf, int batchsize, int bufferNum, EngineBuffer *bufferInfo);
void printFilter(int nDims, int d[], float *outputs);
template <typename T>
void printOutput(int64_t eltCount, T *outputs, EngineBuffer bufferInfo);

extern int eglSetupExtensions(bool is_dgpu);

int main(int argc, char **argv) {

    //----------------Init Cuda Context----------------//
    CUcontext context;
    float* buffer = nullptr;
    cudaMalloc(&buffer, sizeof(float) * 1024);
    cudaFree(buffer);
    cuCtxGetCurrent(&context);


    string socketPath;
    int serviceNumber = 2;
    int iterations = 10;
    vector<InferenceClient*> client;

    // parseSocketPath(argc, argv, socketPath);
    parseServiceNumber(argc, argv, serviceNumber);
    parseIterationNumber(argc, argv, iterations);

    // eglSetupExtensions(0);

    void *pLibs = dlopen("./libclient.so", RTLD_LAZY);
    pCreateEngine         = (PTCreateEngine)            dlsym(pLibs, "CreateEngine");
    pRunEngine            = (PTRunEngine)               dlsym(pLibs, "RunEngine");
    pGetBuffer            = (PTGetBufferOfEngine)       dlsym(pLibs, "GetBufferOfEngine");
    pClientQueryCallback  = (PTClientQueryCallback)     dlsym(pLibs, "ClientQueryCallback");

    if(pCreateEngine == NULL && pRunEngine == NULL) 
        cout<<"Warning: Functions cannot be load."<<endl;

    //----------------Create Cuda Device----------------//
    cudaDeviceCreate();


    string libPath = "./libonnxtrt_orin.so";
    const char* enginePath0 = "./ngp_main_v1.0.0_int8b1_lld.trt";
    const char* enginePath1 = "./ngp_main_v1.0.0_int8b1_kpsodlightning_filter.trt";

    const char* engineConfig = "CudaGraph=True";
    int         engineBatch  = 1;

    unsigned char*         rgbCPUData    = NULL;
    float*                 rgbGPUData    = NULL;
    
    vector<float*>         outputDataCPU(serviceNumber, NULL);
    vector<float*>         outputDataGPU(serviceNumber, NULL);

    EngineBuffer *bufferInfo[32];
    int           bufferNum[32];

    int imgWidth, imgHeight, imgChannel;

    rgbCPUData = stbi_load("./main.png", &imgWidth, &imgHeight, &imgChannel, 3);
    int copyBytesFloat = sizeof(float) * imgHeight * imgWidth * imgChannel;
    cudaMalloc(&rgbGPUData, copyBytesFloat);
    for(int i = 0; i < serviceNumber; i++) {
        outputDataCPU[i] = new float[imgWidth * imgHeight * imgChannel];
        float* tmpGPUPtr = NULL;
        cudaMalloc(&outputDataGPU[i], copyBytesFloat);
    }
    spatial2Channel(rgbCPUData, outputDataCPU[0], imgWidth, imgHeight, imgChannel);
    cudaMemcpy(rgbGPUData, outputDataCPU[0], copyBytesFloat, cudaMemcpyHostToDevice);
    memset(outputDataCPU[0], 0, copyBytesFloat);
    int inputSize, outputSize;

    vector<int>         engineId;
    vector<const char*> engine;
    for(auto i = 0; i < serviceNumber; ++i) {
        const char* curEnginePath = i % 2 == 0 ? enginePath0 : enginePath1;
        engine.push_back(curEnginePath);
        engineId.push_back(pCreateEngine(-1, engine[i], engineConfig));
        pGetBuffer(engineId[i], &bufferInfo[i], &bufferNum[i], NULL);
    }

/*
    //阻塞式
    for(auto i = 0; i < iterations; ++i) {
        if(i % 50 == 0) {
            printf("iterations : %d.\n", i);
        }
        for(auto it = engineId.begin(); it != engineId.end(); it++) {
            pRunEngine(*it, 1, (char*)rgbGPUData, 1, NULL, -1);
        }

        for(auto it = engineId.begin(); it != engineId.end(); it++) {
            pRunEngine(*it, 1, NULL, -1, (char*)outputDataCPU, 0);
        }
    }
*/

    //非阻塞式
    for(auto i = 0; i < iterations; ++i) {
        if(i % 50 == 0) printf("iterations : %d.\n", i);

        //feed data
        for(auto it = engineId.begin(); it != engineId.end(); it++) {
            pRunEngine(*it, 1, (char*)rgbGPUData, 1, NULL, -1);
        }

        //fetch data
        for(auto it = engineId.begin(); it != engineId.end(); it++) {
            int curIdx = distance(engineId.begin(),it);
            if(!curIdx) { 
                //simulate main camera
                pRunEngine(*it, 1, NULL, -1, (char*)outputDataCPU[curIdx], 0);
            } else {
                pRunEngine(*it, 1, NULL, -1, (char*)outputDataGPU[curIdx], -2);
            }
        }
        //Do query Thread
       std::thread queryThread(clientThreadQueryCallback, engineId);
       queryThread.detach();
    }

    for(auto it = engineId.begin(); it != engineId.end(); it++) {
        int curIdx = distance(engineId.begin(),it);
        printf("curIdx: %d.\n", curIdx);
        if(curIdx != 0) {
            cudaMemcpy(outputDataCPU[curIdx], outputDataGPU[curIdx], 8000, cudaMemcpyDeviceToHost);
        }
        printOutBuf(outputDataCPU[curIdx], engineBatch, bufferNum[curIdx], bufferInfo[curIdx]);
    }

}


void clientThreadQueryCallback(vector<int> engineId) {
    
    while (true) {
        bool quit = true;
        for(auto it = engineId.begin(); it != engineId.end(); ++it) {
            quit = quit & pClientQueryCallback(*it);
        }

        if(quit) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

}
                                            

void spatial2Channel(unsigned char* in, float* out, int width, int height, int channel) {
    int pixels = width * height * channel;
    for(int i = 0; i < pixels; ++i) {
        int pixelId   = i / channel;
        int channelId = i % channel;

        int chwHeight = pixelId / width;
        int chwWidth  = pixelId % width;

        int dst_c = ((chwWidth & 1) * 2 + (chwHeight & 1)) * 3 + channelId;
        int dst_y = chwHeight / 2;
        int dst_x = chwWidth  / 2;
        int dst_idx = dst_c * (width / 2) * (height / 2) + dst_y * (width / 2) + dst_x; 

        out[dst_idx] = (in[i] * 1.0f) / 255.f;
    }
}

void printOutBuf(float *pBuf, int batchsize, int bufferNum, EngineBuffer *bufferInfo) {
    int valNum = 0;
    for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;
        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++) eltCount *= bufferInfo[i].d[d];
        printf("[%d]%s.batchId: %d eltCount:%d Data:\n", i, bufferInfo[i].name, nb, eltCount);
        if( strstr(bufferInfo[i].name, "_filter")!=NULL){
          printFilter(bufferInfo[i].nDims, bufferInfo[i].d, pBuf + valNum);
        } else {
          printOutput(eltCount, pBuf + valNum, bufferInfo[i]);
        }
        valNum += eltCount;
      }
    }
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

template <typename T>
void printOutput(int64_t eltCount, T *outputs, EngineBuffer bufferInfo) {
    const int PrintNum = 64;
    printf("the elt count is: %ld\n", eltCount);
    for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) 
        std::cerr << outputs[eltIdx] << "\t, ";
    if (eltCount > PrintNum) {
        std::cerr << " ... ";
        for (int64_t eltIdx = (eltCount - PrintNum); eltIdx < eltCount; ++eltIdx) {
            std::cerr << outputs[eltIdx] << "\t, ";
        }
    }

    std::cerr << std::endl;
}