#include "directShmem.h"
#include "producer_socket_service.h"
#include "onnxtrt.h"

#include <dlfcn.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <chrono>
#include <thread>
#include <string>

using namespace std;


class InferenceServer final {
public:
    InferenceServer(string socketPath){
        socketPath_ = socketPath;
    };

    InferenceServer() = delete;

    ~InferenceServer();

public:
    void Init();
    void DeInit();
    void setWriteMaskValid();
    bool inferIsPermitted();
    bool fetchIsPermitted();
    void runEngine();
    void fetchOutput();
    void syncStream();


public:
    string                 socketPath_;
    char                   libPath_     [MAX_STRING_SIZE];
    char                   enginePath_  [MAX_STRING_SIZE];
    char                   engineConfig_[MAX_STRING_SIZE];
    int                    batchSize_      = 1;
    int                    bufferWidth_, bufferHeight_, bufferChannel_ = 3;

    CUstream              cudaStream_;

    EngineBuffer*          bufferInfo_     = NULL;
    int                    bufferNum_      = 0;

    DirectShmem*           dShmService_    = NULL;
    ProducerSocketService* socketService_  = NULL;
    void*                  eglBuffer_      = NULL;
    void*                  inferData_      = NULL;

    void*                  inferLib_       = NULL;
    int                    engineId_       = -1;
    PTCreateEngine         pCreateEngine_  = NULL;
    PTRunEngine            pRunEngine_     = NULL;
    PTDestoryEngine        pDestoryEngine_ = NULL;
    PTGetBufferOfEngine    pGetBuffer_     = NULL;


    //test params
    int                    inputSize_      = 0;
    int                    outputSize_     = 0;
    bool                   printOutput_    = true;
    bool                   printInput_     = true;

    bool                   needFetched_    = false;
    bool                   fetchNotCalled_ = false;

    key_t                  shmemKey_       = -1;
    int                    shmemId_        = -1;
    int*                   shmemAddr_      = NULL;
};