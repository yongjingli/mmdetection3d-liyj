
#include "helper.h"
#include "inference_server.h"

#include <sys/shm.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <chrono>
#include <thread>

void InferenceCallback(vector<InferenceServer*>& infer);
void printFilter(int nDims, int d[], float *outputs);
void printOutput(InferenceServer* infer, float* outputPtr);
void printInput(InferenceServer* infer, float* inputPtr);
void printNoFilter(int eltCount, float *outputs, EngineBuffer bufferInfo_);

int delete_segment(int seg_id){
    if ((shmctl(seg_id,IPC_RMID,0))==-1){
    std::cout<<" ERROR(C++)with shmctl(IPC_RMID): "<<strerror(errno)<<std::endl;
    return -1;
    }else//on success
        return 0;
}

void cleanSegments() {
    struct shmid_ds shm_info;
    struct shmid_ds shm_segment;
    int max_id = shmctl(0,SHM_INFO,&shm_info);
    if (max_id>=0){
        printf("%d shared memory existing.\n", max_id);
        for (int i=0;i<=max_id;++i) {
                int shm_id = shmctl(i , SHM_STAT , &shm_segment);
                if (shm_id<=0)
                    continue;
                else if (shm_segment.shm_nattch==0){
                    printf("Deleting segments.\n");
                    delete_segment(shm_id);
                }
        }
    }
}

int main(int argc, char **argv) {

    //----------------Init Cuda Context----------------//
    CUcontext context;
    float* buffer = nullptr;
    cudaMalloc(&buffer, sizeof(float) * 1024);
    cudaFree(buffer);
    cuCtxGetCurrent(&context);

    string socketPathBase = "/tmp/client_socket_";
    int serviceNumber = 2;
    int iterations = 10;
    vector<InferenceServer*> infer;

    parseServiceNumber(argc, argv, serviceNumber);
    parseIterationNumber(argc, argv, iterations);

    //----------------Clean Shared Memory----------------//
    cleanSegments();

    for(auto i = 0; i < serviceNumber; i++) {
        string socketPath = socketPathBase + to_string(i);
        InferenceServer* inferTmp = new InferenceServer(socketPath);
        infer.push_back(inferTmp);
    }

    for(auto it = infer.begin(); it != infer.end(); it++) {
        (*it)->Init();
    }

    for(auto i = 0; i < iterations; ++i) {
        
        for(auto it = infer.begin(); it != infer.end(); it++) {
           if((*it)->inferIsPermitted()) { 
                printf("iterations : %d, find valid -- Inference.\n", i);
                (*it)->runEngine();
            } else {
                // printf("iterations : %d, not find valid -- Inference.\n", i);
            }
        }

        for(auto it = infer.begin(); it != infer.end(); it++) {
           if((*it)->fetchIsPermitted()) { 
               printf("iterations : %d, Fetching Output -- Inference.\n", i);
                (*it)->fetchOutput();
                (*it)->setWriteMaskValid();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    for(auto it = infer.begin(); it != infer.end(); it++) {
        (*it)->DeInit();

        delete (*it);
    }


    // printf("start sleep...\n");
    // sleep(8);

    // //----------------Deinit----------------//
    // for(auto it = infer.begin(); it != infer.end(); it++) {
    //     (*it)->pDestoryEngine((*it)->engineId_);
    //     dlclose((*it)->inferLib);
    // }


}

void InferenceCallback(vector<InferenceServer*>& infer) {
    // for(auto it = infer.begin(); it != infer.end(); ++it) {
    //     if((*it)->dShmService_->checkReadMaskStatusValid((*it)->cudaStream_)) { 
    //         printf("find valid -- Inference.\n");
    //         // if ((*it)->printInput) {
    //         //     (*it)->printInput = false;
    //         //     printInput((*it), (*it)->dShmService_->getDataPtr());
    //         // }
    //         (*it)->pRunEngine_((*it)->engineId_, 1, reinterpret_cast<char*>((*it)->dShmService_->getDataPtr()), 1, nullptr, -2);
    //     }
    // }

    // for(auto it = infer.begin(); it != infer.end(); ++it) {
    //     if((*it)->dShmService_->checkReadMaskStatusValid((*it)->cudaStream_)) { 
    //         printf("Fetching Data.\n");
    //         (*it)->pRunEngine_((*it)->engineId_, 1, nullptr, -1, reinterpret_cast<char*>((*it)->dShmService_->getDataPtr()), -2);
    //         if((*it)->printOutput_) {
    //             (*it)->printOutput_ = false;
    //             printOutput((*it), (*it)->dShmService_->getDataPtr());
    //         }
    //         // (*it)->dShmService_->setWriteMaskValid((*it)->cudaStream_);
    //     }
    // }
}


void printOutput(InferenceServer* infer, float* outputPtr) {
//     printf("Start printing output.\n");
//     int outputSize = infer->imgChannel_ * infer->imgHeight_ * infer->imgWidth_;
//     float* tmpBuffer = new float[outputSize];
//     for (int i = 0; i < infer->bufferNum_; ++i) {
//         if (0 == infer->bufferInfo_[i].nBufferType) 
//             continue;
//         int eltCount = 1;
//         for (int d = 0; d < infer->bufferInfo_[i].nDims; ++d) 
//             eltCount *= infer->bufferInfo_[i].d[d];
//         cudaMemcpy(tmpBuffer, outputPtr, sizeof(float) * eltCount, cudaMemcpyDeviceToHost);
//         printf("%s:\n", infer->bufferInfo_[i].name);
//         if( strstr(infer->bufferInfo_[i].name, "_filter") != NULL){
//           printFilter(infer->bufferInfo_[i].nDims, infer->bufferInfo_[i].d, tmpBuffer);
//         } else {
//           printNoFilter(eltCount, tmpBuffer, infer->bufferInfo_[i]);
//         }
//         outputPtr += eltCount;
//     }

//     delete [] tmpBuffer;
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

void printNoFilter(int eltCount, float *outputs, EngineBuffer bufferInfo_) {
    int printNum = 64;
    for(int i = 0; i < printNum && i < eltCount; ++i)
        cout<<outputs[i]<<"\t, ";
        // printf("%.3lf\t", outputs[i]);

    printf("...\t");

    for(int i = (eltCount - printNum); i < eltCount; ++i) 
        cout<<outputs[i]<<"\t, ";
        // printf("%.3lf\t", outputs[i]);

    printf("\n");
}