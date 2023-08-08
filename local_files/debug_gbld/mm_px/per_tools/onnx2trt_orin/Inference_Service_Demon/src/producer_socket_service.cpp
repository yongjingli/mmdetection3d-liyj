#include "producer_socket_service.h"

int ProducerSocketService::creatUnixSocket(string socketPath) {
    if (connect_ == -1) {
        socketPath_ = socketPath;
        return creatUnixSocket();
    }
    else 
        return connect_;
}


int ProducerSocketService::creatUnixSocket(void) {

    listen_ = socket(PF_UNIX, SOCK_STREAM, 0); //创建流socket
    if (listen_ < 0) {
        cout<<"Socket creat failed."<<endl;
        return -1;
    }

    unlink(socketPath_.c_str());

    memset(&sockAddr_, 0, sizeof(struct sockaddr_un));
    sockAddr_.sun_family = AF_UNIX;
    strncpy(sockAddr_.sun_path,
            socketPath_.c_str(),
            sizeof(sockAddr_.sun_path)-1);

    if (bind(listen_,
             (const struct sockaddr*)&sockAddr_,
             sizeof(struct sockaddr_un))) {
        cout<<"Socket bind error."<<endl;
        return -1;
    }

    if (listen(listen_, 1)) {
        cout<<"Socket listen error."<<endl;
        return -1;
    }

    connect_ = accept(
                    listen_,
                    (struct sockaddr *)&connectAddr_,
                    &connectAddrLen_);

    close(listen_);
    // unlink(socketPath_.c_str());
    if (connect_ < 0) {
        cout<<"Socket accept error."<<endl;
        return -1;
    }

    return connect_;

}


int ProducerSocketService::recieveEngineInfo(char* libPath, char* enginePath, char* engineConfig) {
    struct msghdr msg;
    struct iovec iov[3];
    char ctrl_buf[CMSG_SPACE(sizeof(int))];
    struct cmsghdr *cmsg;
    void *data;
    int recvfd;

    memset(&msg, 0, sizeof(msg));
    iov[0].iov_base = libPath;
    iov[0].iov_len  = MAX_STRING_SIZE;
    iov[1].iov_base = enginePath;
    iov[1].iov_len  = MAX_STRING_SIZE;
    iov[2].iov_base = engineConfig;
    iov[2].iov_len  = MAX_STRING_SIZE;
    msg.msg_iov = iov;
    msg.msg_iovlen = 3;

    msg.msg_control = ctrl_buf;
    msg.msg_controllen = sizeof(ctrl_buf);

    if (recvmsg(connect_, &msg, 0) <= 0) {
        printf("%s: rec config vmsg failed\n", __func__);
        return -1;
    }
    

    cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg) {
        printf("%s: NULL message header\n", __func__);
        return -1;
    }
    if (cmsg->cmsg_level != SOL_SOCKET) {
        printf("%s: Message level is not SOL_SOCKET\n", __func__);
        return -1;
    }
    if (cmsg->cmsg_type != SCM_RIGHTS) {
        printf("%s: Message type is not SCM_RIGHTS\n", __func__);
        return -1;
    }

    data = CMSG_DATA(cmsg);

    return 1;
}

int ProducerSocketService::recieveShmemKey(key_t& shmKey) {
    struct msghdr msg;
    struct iovec iov[1];

    memset(&msg, 0, sizeof(msg));

    iov[0].iov_base = &shmKey;
    iov[0].iov_len  = sizeof(key_t);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    if (recvmsg(connect_, &msg, 0) <= 0) {
        printf("%s: rec shared memory id failed\n", __func__);
        return -1;
    }

    return 1;
}

int ProducerSocketService::sendEngineIOSize(int& inputSize, int& outputSize) {
    uint8_t msg[8];
    reinterpret_cast<int*>(msg)[0] = inputSize;
    reinterpret_cast<int*>(msg)[1] = outputSize;

    if (send(connect_, msg, 8, 0) <= 0) {
        printf("%s: send engine IO size msg failed\n", __func__);
        return -1;
    }

    return 1;
}

int ProducerSocketService::sendEngineBufferNum(int& bufferNum) {
    if (send(connect_, &bufferNum, sizeof(int), 0) <= 0) {
        printf("%s: send engine buffer number msg failed\n", __func__);
        return -1;
    }

    return 1;
}

int ProducerSocketService::sendEngineBufferInfo(EngineBuffer *engineBuffer, int bufferNum) {

    if (send(connect_, engineBuffer, bufferNum * sizeof(EngineBuffer), 0) <= 0) {
        printf("%s: send engine buffer information msg failed\n", __func__);
        return -1;
    }

    return 1;
}

int ProducerSocketService::sendEngineBufferNames(EngineBuffer *engineBuffer, int bufferNum) {
    for(auto i = 0; i < bufferNum; ++i) {
        char tmpCharArray[MAX_STRING_SIZE];
        strcpy(tmpCharArray, engineBuffer[i].name);
        if (send(connect_, tmpCharArray, MAX_STRING_SIZE, 0) <= 0) {
            printf("%s: send engine name information msg failed\n", __func__);
            return -1;
        }
    }

    return 1;
}
