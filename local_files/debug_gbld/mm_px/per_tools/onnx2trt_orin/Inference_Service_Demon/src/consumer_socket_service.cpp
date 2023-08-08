#include "consumer_socket_service.h"

int ConsumerSocketService::creatUnixSocket(string socketPath) {
    if(sock_ == -1) {
        socketPath_ = socketPath;
        return creatUnixSocket();
    } else
        return sock_;

}


int ConsumerSocketService::creatUnixSocket(void) {
    sock_ = socket(PF_UNIX, SOCK_STREAM, 0);
    if(sock_ < 0) {
        cout<<"Socket creat failed."<<endl;
        return -1;
    }

    auto flags  = fcntl(sock_,F_GETFL,0);

    fcntl(sock_,F_SETFL,flags&~O_NONBLOCK); 

    memset(&sockAddr_, 0, sizeof(struct sockaddr_un));
    sockAddr_.sun_family = AF_UNIX;
    strncpy(sockAddr_.sun_path,
            socketPath_.c_str(),
            sizeof(sockAddr_.sun_path)-1);
    return sock_;
}

int ConsumerSocketService::connectWithProducerSocket(void) {
    while (connect(sock_,
                (const struct sockaddr*)&sockAddr_,
                sizeof(struct sockaddr_un))) {
        if(waitLoop_ < 60) {
            if(!waitLoop_)
                cout<<"Waiting for Producer at "<<socketPath_<<endl;
            else
                cout<<".";
            fflush(stdout);
            sleep(1);
            waitLoop_ ++;
        } else {
            cout<<"Waiting time out."<<endl;
            return -1;
        }
    }

    if(waitLoop_)   cout<<" "<<endl;
    cout<<"Got Connection."<<endl;

    return sock_;
}

int ConsumerSocketService::sendShmemKey(key_t& shmKey) {
    struct msghdr msg;
    struct iovec iov[1];
    memset(&msg, 0, sizeof(msg));

    iov[0].iov_len  = sizeof(key_t);   
    iov[0].iov_base = (void*)(&shmKey);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    int res = sendmsg(sock_, &msg, 0);
    if(res <= 0) {
        printf("%s: send shared memory key failed.\n", __func__);
        return -1;
    }

    return 1;
}

int ConsumerSocketService::sendEngineInfo(char* libPath, char* enginePath, char* engineConfig) {
    struct msghdr msg;
    struct iovec iov[3];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr *cmsg = NULL;
    void *data;
    int res;
    memset(&msg, 0, sizeof(msg));
 
    iov[0].iov_len  = MAX_STRING_SIZE;  
    iov[0].iov_base = (void*)libPath;  
    iov[1].iov_len  = MAX_STRING_SIZE;  
    iov[1].iov_base = (void*)enginePath; 
    iov[2].iov_len  = MAX_STRING_SIZE;  
    iov[2].iov_base = (void*)engineConfig; 
    msg.msg_iov = iov;
    msg.msg_iovlen = 3;

    memset(ctrl_buf, 0, sizeof(ctrl_buf));
    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    data = CMSG_DATA(cmsg);
    *(int *)data = 0;

    msg.msg_controllen = cmsg->cmsg_len;
    res = sendmsg(sock_, &msg, 0);
    if(res <= 0) {
        printf("%s: sendmsg failed", __func__);
        return -1;
    }

    return 0;
}

int ConsumerSocketService::recieveEngineIOSize(int& inputSize, int& outputSize) {
    uint8_t msg[8];
     if (recv(sock_, msg, 8, 0) <= 0) {
        printf("%s: recieve IO size failed\n", __func__);
        return -1;
    }

    inputSize = reinterpret_cast<int*>(msg)[0];
    outputSize = reinterpret_cast<int*>(msg)[1];

    printf("The engine IO Size is %d, %d.\n", inputSize, outputSize);

    return 1;
}

int ConsumerSocketService::recieveEngineBufferNum(int& bufferNum) {
    if (recv(sock_, &bufferNum, sizeof(int), 0) <= 0) {
        printf("%s: recieve buffer num failed\n", __func__);
        return -1;
    }
    printf("Client lib: Engine Buffer Num is %d.\n", bufferNum);
    return 1;
}

int ConsumerSocketService::recieveEngineBufferInfo(EngineBuffer **engineBuffer, int bufferNum) {
    
    *engineBuffer = (EngineBuffer*)malloc(sizeof(EngineBuffer) * bufferNum);

    if (recv(sock_, *engineBuffer, sizeof(EngineBuffer) * bufferNum, 0) <= 0) {
        printf("%s: recieve buffer info msg failed\n", __func__);
        return -1;
    }

    printf("Client lib:input batch size is %d.\n", (*engineBuffer)[0].nMaxBatch);

    return 1;
}

int ConsumerSocketService::recieveEngineBufferNames(EngineBuffer **engineBuffer, int bufferNum, char bufferName[][MAX_STRING_SIZE]) {
    
    for(auto i = 0; i < bufferNum; ++i) {
        if (recv(sock_, bufferName[i], MAX_STRING_SIZE, 0) <= 0) {
            printf("%s: recieve buffer info msg failed\n", __func__);
            return -1;
        }
        (*engineBuffer)[i].name = bufferName[i];
        // printf("The engine buffer name is %s.\n", bufferName[i]);
    }


    return 1;
}

void ConsumerSocketService::DeInit() {
    close(sock_);
}
