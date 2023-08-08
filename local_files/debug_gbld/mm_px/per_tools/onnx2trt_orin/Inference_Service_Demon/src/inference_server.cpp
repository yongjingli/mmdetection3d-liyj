#include "inference_server.h"

void setWriteMaskCallback(void* userData) {
    InferenceServer* inferServiceTmp = (InferenceServer*) userData;
    inferServiceTmp->dShmService_->setWriteMaskValid();
    inferServiceTmp->needFetched_ = false;

}

void InferenceServer::Init() {
    //----------------Create Socket Service----------------//
    cout<<"The socket path is "<<socketPath_<<endl;
    socketService_ = new ProducerSocketService(socketPath_);
    socketService_->creatUnixSocket();
    //----------------Recieve EGL FD from infers----------------//
    socketService_->recieveEngineInfo(libPath_, enginePath_, engineConfig_);
    cout<<"library path is: "<<libPath_<<endl;
    cout<<"engine path is: "<<enginePath_<<endl;
    cout<<"engine config is: "<<engineConfig_<<endl;

    //---------------Creat Inference Engine----------------//
    inferLib_ = dlopen(libPath_, RTLD_LAZY);
    if (NULL == inferLib_) {
        cout<<"Can not open inference library."<< libPath_<<endl;;
    }

    pCreateEngine_  = (PTCreateEngine)     dlsym(inferLib_, "CreateEngine");
    pRunEngine_     = (PTRunEngine)        dlsym(inferLib_, "RunEngine");
    pDestoryEngine_ = (PTDestoryEngine)    dlsym(inferLib_, "DestoryEngine");
    pGetBuffer_     = (PTGetBufferOfEngine)dlsym(inferLib_, "GetBufferOfEngine");

    if (NULL == pCreateEngine_  || NULL == pCreateEngine_ || 
        NULL == pDestoryEngine_ || NULL == pGetBuffer_) {  
        cout<<"Can not load Function from " << libPath_<<endl; 
        dlclose(inferLib_);
    } else {
        cout<<"Load functions from"<< libPath_<<" successfully." <<endl;
    }

    engineId_ = pCreateEngine_(-1, enginePath_, engineConfig_);
    // cout<<"The engine idx is."<< engineId_<<endl;

    //---------------Get Engine Buffer----------------//
    int paddingWidth = -1;
    void* stream;
    pGetBuffer_(engineId_, &bufferInfo_, &bufferNum_, &stream);
    cudaStream_ = (CUstream) stream;
    for(int i = 0; i < bufferNum_; ++i) {
        EngineBuffer &buffInfo = bufferInfo_[i];
        if(buffInfo.nBufferType > 0) { //outputBuffer
            outputSize_ += buffInfo.nBufferSize;
        } else {    //inputBuffer
            inputSize_ += buffInfo.nBufferSize;
            paddingWidth = paddingWidth == -1 ? buffInfo.d[2] : paddingWidth;
        }
    }

    socketService_->sendEngineIOSize(inputSize_, outputSize_);
    socketService_->sendEngineBufferNum(bufferNum_);
    socketService_->sendEngineBufferInfo(bufferInfo_, bufferNum_);
    socketService_->sendEngineBufferNames(bufferInfo_, bufferNum_);

    int allocatePage = (inputSize_ + outputSize_) / (4 * 1024) + 2;
    int allocateSize = allocatePage * 4 * 1024;
    printf("allocate page size is %d.\n", allocateSize);
    int testSize = 4 * 1024 * 1300;
    socketService_->recieveShmemKey(shmemKey_);
    printf("Shmem Key is %d.\n", shmemKey_);
    //----------------Get Shared Memory----------------//
    shmemId_ = shmget(shmemKey_, 0, 0666);
	if(shmemId_ == -1)  cout << "shmget failed at infer server initialization." << endl;
    else                cout << "The shared memory id is "<<shmemId_<<endl;
    // shmctl(shmemId_, IPC_RMID, NULL);
    int* shmemAddr_ = (int*)shmat(shmemId_, NULL, 0);

    cudaHostRegister(shmemAddr_, allocateSize, cudaHostRegisterMapped);
    int* shmemDevPtr = NULL;
    cudaHostGetDevicePointer(&shmemDevPtr, shmemAddr_, 0);
    //----------------Producer Prepare directShmem Service----------------//
    dShmService_ = new DirectShmem(((float*)shmemDevPtr) + 1, shmemAddr_);

    printf("\n\n");

}

void InferenceServer::syncStream() {
    cudaStreamSynchronize(cudaStream_);
}

void InferenceServer::setWriteMaskValid() {
    // dShmService_->setWriteMaskValid();
    cuLaunchHostFunc(cudaStream_, setWriteMaskCallback, this);
}

bool InferenceServer::inferIsPermitted() {
    return dShmService_->checkReadMaskStatusValid() && !needFetched_;
}

bool InferenceServer::fetchIsPermitted() {
    return dShmService_->checkReadMaskStatusValid() && needFetched_ && fetchNotCalled_; 
}

void InferenceServer::runEngine() {
    needFetched_ = true;
    fetchNotCalled_ = true;
    pRunEngine_(engineId_, batchSize_, (char*)dShmService_->getDataPtr(), 1, nullptr, -2);
}

void InferenceServer::fetchOutput() {
    if(fetchNotCalled_) {
        fetchNotCalled_ = false;
        pRunEngine_(engineId_, batchSize_, nullptr, -1, (char*)dShmService_->getDataPtr(), -2);
    }
    
}

void InferenceServer::DeInit() {
    pDestoryEngine_(engineId_);
    dlclose(inferLib_);
}

InferenceServer::~InferenceServer() {
    // printf("Doing Deconstructor.\n");

    cudaHostUnregister(shmemAddr_);

    if(!shmemAddr_) {
        // printf("Inference Server Detaching.\n");
        if(shmdt(shmemAddr_) == -1) {
            // printf("Detaching shared memory Failed.\n");
        }
    }
        

    if(shmemId_ != -1) {
        // printf("Removing shared memory.\n");
        if(shmctl(shmemId_, IPC_RMID, NULL) == -1) {
            // printf("Removing shared memory Failed.\n");
        }
    }
        
}

