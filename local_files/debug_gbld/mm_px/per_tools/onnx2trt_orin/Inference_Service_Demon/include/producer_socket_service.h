#include "socket_service.hpp"
#include "onnxtrt.h"

#include <string>
#include <iostream>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>


class ProducerSocketService final : public SocketService {
public:
    // using SocketService::SocketService;
    ProducerSocketService(string socketPath):SocketService(socketPath) {};
    ProducerSocketService();

    virtual int creatUnixSocket(string socketPath);
    virtual int creatUnixSocket(void);


    int  recieveEngineInfo(char* libPath, char* enginePath, char* engineConfig);
    int  recieveShmemKey(key_t& shmKey);

    int  sendEngineIOSize(int& inputSize, int& outputSize);
    int  sendEngineBufferNum(int& bufferNum);
    int  sendEngineBufferInfo(EngineBuffer *engineBuffer, int bufferNum);
    int  sendEngineBufferNames(EngineBuffer *engineBuffer, int bufferNum);

    int  getConnect(){return connect_;}

private:

    int listen_ = -1, connect_ = -1;
    struct sockaddr_un connectAddr_;
    socklen_t connectAddrLen_ = 0;
};  