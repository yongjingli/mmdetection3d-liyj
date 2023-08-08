#include "socket_service.hpp"
#include "onnxtrt.h"

#include <string>
#include <iostream>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>


class ConsumerSocketService final : public SocketService {
public:
    // using SocketService::SocketService;
    ConsumerSocketService(string socketPath):SocketService(socketPath) {};
    ConsumerSocketService();

    virtual int creatUnixSocket(string socketPath);
    virtual int creatUnixSocket(void);

    int connectWithProducerSocket(void);

    int sendEngineInfo(char* libPath, char* enginePath, char* engineConfig);
    int sendShmemKey(key_t& shmKey);

    int recieveEngineIOSize(int& inputSize, int& outputSize);
    int recieveEngineBufferNum(int& bufferNum);
    int recieveEngineBufferInfo(EngineBuffer **engineBuffer, int bufferNum);
    int recieveEngineBufferNames(EngineBuffer **engineBuffer, int bufferNum, char bufferName[][MAX_STRING_SIZE]);
    int getSockId(void) { return sock_; };
    
    void DeInit();

private:

    int sock_= -1;
    int waitLoop_ = 0;
};