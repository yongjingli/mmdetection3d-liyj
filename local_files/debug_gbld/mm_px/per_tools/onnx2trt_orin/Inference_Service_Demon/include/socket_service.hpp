#include <string>
#include <iostream>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;

#define MAX_STRING_SIZE 200

class SocketService {
    
public:
    SocketService(string socketPath) {socketPath_ = socketPath;};
    SocketService();

    virtual int creatUnixSocket(string socketPath) = 0;
    virtual int creatUnixSocket(void) = 0;

protected:

    string socketPath_;
    struct sockaddr_un sockAddr_;


};
