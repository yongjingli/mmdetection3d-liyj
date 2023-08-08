#include "directShmem.h"
#include "onnxtrt.h"
#include "consumer_socket_service.h"
#include "common.h"

#include <chrono>
#include <thread>
#include <string>
#include <iostream>               // std::cout
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;

#define MAX_CLIETN_NUM 32
#ifdef __cplusplus
extern "C" {
#endif
// 2022.05.16 v0.0.1 Finish the first version of inference server. The mask is passed through CUDA buffer of DirectEGL   
//                   between different processes. 
// 2022.05.17 v0.0.2 Finish the second version of inference server. The mask is passed through shared memory between
//                   different processes.
// 2022.05.18 v0.0.3 Finish the third version of inference server. The non-blocking thread query is finished and validated.


#define CLIENT_VERSION_STRING "V0.0.3.orin"
#define CLIENT_CHECK 1
#define CLIENT_OK 0
#define CLIENT_GENEREL_ERR -1    // General error, no detail yet.
#define CLIENT_INVALID_ERR -2    // invalid resource/ID, not free or idle.
#define CLIENT_PARAMETER_ERR -3  // Error paramenter/argument.
#define CLIENT_IO_ERR -4         // Error when read/write buffer or file.
#define CLIENT_CUDA_ERR -5       // Error of CUDA.  

bool ClientQueryCallback(int ch);

typedef bool (*PTClientQueryCallback)(int ch);

#ifdef __cplusplus
}
#endif


typedef void (*pCallback)(void*);

class InferenceClient {
public:
    InferenceClient(){};

    ~InferenceClient();

    void   Init(int channelIdx, int shmemKey, const char* socketPath, 
                const char* libPath, const char* enginePath, const char* engineConfig);
    void   registerCallbackFunction(pCallback callbackFun, void* userData);
    bool   callBackNotRun(){ return pCallbackFun_ != NULL && callbackNotRun_; }
    bool   outputFetched() {return isFetched_; }
    void   runCallbackFuntion();
    void   setReadMaskValid();
    bool   writeIsPermitted();
    float* getEGLDataBuffer();

    bool   isInitiated() {return isInitiated_;};
    bool   isBlockedMode() {return isBlockedMode_;};

    void   runEngine(int batch, char* inputData, int inputType, char* outputData, int outputType);
    int    getEngineBuffer(EngineBuffer **pBufferInfo, int *pNum, void **pStream);
    void   fetchDataCallback();

    void   DeInit();

    // void   notifyCV();

public:
    //Engine Information
    string                 socketPath_;
    char                   libPath_     [MAX_STRING_SIZE];
    char                   enginePath_  [MAX_STRING_SIZE];
    char                   engineConfig_[MAX_STRING_SIZE];

    CUstream               cudaStream_;

    int                    channelIdx_     = -1;
    int                    outputType_     = 0;
    int                    inputSize_      = -1;
    int                    outputSize_     = -1;
    bool                   isBlockedMode_  = true;

    EngineBuffer*          bufferInfo_     = NULL;
    int                    bufferNum_      = 0;
    char                   bufferName_[ONNXTRT_MAX_BUFFERNUM][MAX_STRING_SIZE];

    char*                 customerOutputPtr_ = NULL;

    //Engine Status
    bool                   isInitiated_      = false;
    bool                   isFetched_        = false;
    bool                   callbackNotRun_   = true;

    bool                   writeInputFinish_ = false;

    //Inference Services
    DirectShmem*           dShmService_      = NULL;
    ConsumerSocketService* socketService_    = NULL;

    //Call Back Function
    pCallback              pCallbackFun_     = NULL;
    void*                  userData_         = NULL;

    key_t                  shmemKey_         = -1;
    int                    shmemId_          = -1;
    int*                   shmemAddr_        = NULL;

};

