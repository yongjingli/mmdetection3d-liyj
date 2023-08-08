#include "inference_client.h"

void setReadMaskCallback(void* userData) {
    InferenceClient* inferClient = (InferenceClient*) userData;
    inferClient->dShmService_->setReadMaskValid();
    inferClient->writeInputFinish_ = true;
}


void InferenceClient::Init(int channelIdx, int shmemKey, const char* socketPath,
                            const char* libPath, const char* enginePath, const char* engineConfig) {
    isInitiated_ = true;              
    channelIdx_  = channelIdx;

    //----------------Init Cuda Context----------------//
    float* buffer = nullptr;
    cudaMalloc(&buffer, sizeof(float) * 1024);
    cudaFree(buffer);

    cuStreamCreate(&cudaStream_, CU_STREAM_NON_BLOCKING);

    //----------------Init Engine Information------------------//
    socketPath_   = socketPath;
    strcpy(libPath_,      libPath);
    strcpy(enginePath_,   enginePath);
    strcpy(engineConfig_, engineConfig);
    shmemKey_ = shmemKey;

    //----------------Create Socket Service------------------//
    socketService_ = new ConsumerSocketService(socketPath_);
    socketService_->creatUnixSocket();
    socketService_->connectWithProducerSocket();
    //----------------Create EGL Service------------------//
    socketService_->sendEngineInfo(libPath_, enginePath_, engineConfig_);
    
    socketService_->recieveEngineIOSize(inputSize_, outputSize_);
    socketService_->recieveEngineBufferNum(bufferNum_);
    socketService_->recieveEngineBufferInfo(&bufferInfo_, bufferNum_);
    socketService_->recieveEngineBufferNames(&bufferInfo_, bufferNum_, bufferName_);
    //----------------Create Shared Memory------------------//
    int allocatePage = ((inputSize_ + outputSize_) / (4 * 1024)) + 2;
    int allocateSize = allocatePage * 4 * 1024;
    int testSize = 4 * 1024 * 1300;
    // shmemKey_ = 5680;
    shmemId_ = shmget(shmemKey_, allocateSize , IPC_CREAT | IPC_EXCL | 0666);
    // shmctl(shmemId_, IPC_RMID, NULL);
	if(shmemId_ == -1)  cout << "Error: shmget failed at client initialization." << endl;
    else                cout << "The shared memory id is "<<shmemId_<<endl;
	
	int* shmemAddr_  = (int*)shmat(shmemId_, NULL, 0);
    cudaHostRegister(shmemAddr_, allocateSize, cudaHostRegisterMapped);
    int* shmemDevPtr = NULL;
    cudaHostGetDevicePointer(&shmemDevPtr, shmemAddr_, 0);

    socketService_->sendShmemKey(shmemKey_);

    dShmService_ = new DirectShmem(((float*)shmemDevPtr) + 1, shmemAddr_);


    //----------------Set Write Mask Valid------------------//
    dShmService_->setWriteMaskValid();
    // printf("Client Initialization Finish.\n");
}

void InferenceClient::setReadMaskValid() {
    dShmService_->setReadMaskValid();
}

bool InferenceClient::writeIsPermitted() {
    return dShmService_->checkWriteMaskStatusValid();
}

float* InferenceClient::getEGLDataBuffer() {
    return dShmService_->getDataPtr();
}


void InferenceClient::runEngine(int batch, char* inputData, int inputType, char* outputData, int outputType) {
    if (inputType == 1 && nullptr != inputData) {
        while(!dShmService_->checkWriteMaskStatusValid()) {

                std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        writeInputFinish_ = false;
        isFetched_        = false;
        cudaMemcpyAsync(dShmService_->getDataPtr(), inputData, inputSize_ * batch, cudaMemcpyDeviceToDevice, cudaStream_);
        cuLaunchHostFunc(cudaStream_, setReadMaskCallback, this);
    }

    if(outputData != NULL) {
        if(outputType >= 0) {
            while(!dShmService_->checkWriteMaskStatusValid() || !writeInputFinish_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            cudaMemcpyAsync(outputData, dShmService_->getDataPtr(), outputSize_ * batch, cudaMemcpyDeviceToHost, cudaStream_);
            isFetched_ = true;
        } else {
            customerOutputPtr_ = outputData;
        }
    }

    if (outputType >= 0)   cudaStreamSynchronize(cudaStream_);
    
}

int InferenceClient::getEngineBuffer(EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
    if (NULL == pNum || NULL == pBufferInfo) return CLIENT_PARAMETER_ERR;

    if (NULL == pStream)    cudaStreamSynchronize(cudaStream_);
    else                    *pStream = cudaStream_;
    
    *pNum = bufferNum_;
    *pBufferInfo = bufferInfo_;

    return 0;

}

void InferenceClient::fetchDataCallback() {
    cudaMemcpyAsync(customerOutputPtr_, dShmService_->getDataPtr(), outputSize_, cudaMemcpyDeviceToDevice, cudaStream_);
    cudaStreamSynchronize(cudaStream_);
    isFetched_ = true;
}

void InferenceClient::registerCallbackFunction(pCallback callbackFun, void* userData) {
    pCallbackFun_ = callbackFun;
    userData_     = userData;
}

void InferenceClient::runCallbackFuntion() {
    pCallbackFun_(userData_);
    callbackNotRun_ = false;
}

//TODO: DeInit
void InferenceClient::DeInit() {
    socketService_->DeInit();
    if(!bufferInfo_)    free(bufferInfo_);
}

InferenceClient::~InferenceClient() {
    cudaHostUnregister(shmemAddr_);
    if(!shmemAddr_) {
        printf("Net Work Client Detaching.\n");
        shmdt(shmemAddr_);
    }
}


//Client Interface


static string default_socket_path = "/tmp/client_socket";
static int    socket_counter      = -1;
static string default_lib_path    = "./libonnxtrt_orin.so";
static int    default_shmem_idx   = 5678;

static InferenceClient infer_client       [MAX_CLIETN_NUM];
static InferenceClient infer_client_second[MAX_CLIETN_NUM];

extern "C" int CreateEngine(int ch, const char* engine_filename, const char* engine_config) {
    printf("libclient creating engine.\n");
    //----------------First Check Idle Channel------------------//
    if(ch >= 0 && ch < MAX_CLIETN_NUM) {
        if(infer_client[ch].isInitiated())
            return CLIENT_INVALID_ERR;
    } else if(ch == -1) {
        for (int i = MAX_CLIETN_NUM - 1; i >= 0; i--) {
            if (!infer_client[i].isInitiated()) {
                ch = i;
                break;
            }
        }

        if (-1 == ch) return CLIENT_INVALID_ERR;
    } else {
        return CLIENT_INVALID_ERR;
    }

    //----------------Create Second Engine ------------------//
    const unsigned short TRTMAGIC[2] = {0x7470, 0x7472};  // TRT 6 & 7

    size_t size = 0;
    size_t name_length = strnlen(engine_filename, NAME_MAX +1);
    char   *p_first  = nullptr;
    char   *p_second = nullptr;

    if (name_length < NAME_MAX && name_length > 4 && 0 == strncasecmp(&engine_filename[name_length - 4], ".trt", 4)) {
        char full_filename[NAME_MAX];  // strtok will replace ','
        strncpy(full_filename, engine_filename, sizeof(full_filename));
        // check if has two trt file, split by ','
        p_first = strtok(full_filename, ",");
        p_second = strtok(NULL, ",");
        if (p_first && p_second) {
            socket_counter ++;
            string socket_path = default_socket_path + "_" + to_string(socket_counter);
            default_shmem_idx += 1;
            infer_client_second[ch].Init(ch, default_shmem_idx, socket_path.c_str(), 
                                            default_lib_path.c_str(), p_second, engine_config);
        }
            
    } else {
        printf("CreateEngine failed from %-60.60s \n", engine_filename);
        return CLIENT_IO_ERR;
    }

    //----------------Create First Engine ------------------//
    socket_counter ++;
    string socket_path = default_socket_path + "_" + to_string(socket_counter);
    default_shmem_idx += 1;
    infer_client[ch].Init(ch, default_shmem_idx, socket_path.c_str(), default_lib_path.c_str(), p_first, engine_config);

    return ch;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **buffer_info, int *p_num, void **p_stream) {
    return infer_client[ch].getEngineBuffer(buffer_info, p_num, p_stream);
}


extern "C" int RunEngine(int ch, int batch, char *input_data, int input_type, char *out_data, int out_type) {
    if(!infer_client[ch].isInitiated()) 
        return CLIENT_INVALID_ERR;

    infer_client[ch].isBlockedMode_ = out_type >= 0 ? true : false;
    infer_client[ch].callbackNotRun_ = true;

    if(infer_client[ch].bufferInfo_[0].nMaxBatch != batch && infer_client_second[ch].isInitiated()) {
        infer_client_second[ch].runEngine(batch, input_data, input_type, out_data, out_type);
        return CLIENT_OK;
    }

    infer_client[ch].runEngine(batch, input_data, input_type, out_data, out_type);

    return CLIENT_OK;
}

extern "C" int DestoryEngine(int ch) {
    if (!infer_client[ch].isInitiated()) {
        return -2;
    }
    infer_client[ch].DeInit();


    return CLIENT_OK;
}

extern "C" bool ClientQueryCallback(int ch) {
    //skip if main or fetched
    if(infer_client[ch].isBlockedMode()) {

        if(infer_client[ch].callBackNotRun())
            infer_client[ch].runCallbackFuntion();

        return true;
    } else {
        if(infer_client[ch].outputFetched()) {
            return true;
        } else {
            if(infer_client[ch].writeIsPermitted()) {
                infer_client[ch].fetchDataCallback();

                if(infer_client[ch].callBackNotRun())
                    infer_client[ch].runCallbackFuntion();

                return true;
            } else {
                return false;
            }
        }
    }

}

//These two functions are for test multi model
extern "C" void* AllocateSpaceGPU(size_t bytes, int num) {
    void* gpuTmpPtr = nullptr;
    cudaMalloc(&gpuTmpPtr, bytes * num);

    return gpuTmpPtr;
}

extern "C" void MemcpyHost2DeviceGPU(void* dst, void* src, size_t bytes, int num) {
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    cudaMemcpyAsync(dst, src, bytes * num, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


int TRT_DEBUGLEVEL=1;
static __attribute__((constructor)) void lib_init(void) {
    {
        char *val = getenv("TRT_DEBUGLEVEL");
        if (NULL != val) {
        TRT_DEBUGLEVEL = atoi(val);
        }
    }
    printf("Load client infernce lib %s built@%s %s DebugLevel=%d\n", CLIENT_VERSION_STRING, __DATE__, __TIME__, TRT_DEBUGLEVEL);
}

static __attribute__((destructor)) void lib_deinit(void) {
    printf("Unload client infernce lib %s built@%s %s\n", CLIENT_VERSION_STRING, __DATE__, __TIME__);
}

