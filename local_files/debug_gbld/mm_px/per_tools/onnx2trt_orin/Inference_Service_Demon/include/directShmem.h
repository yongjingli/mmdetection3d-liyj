#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#define WRITEVALIDMASK 0xAAAAAAAA
#define WRITEINVALIDMASK 0xBBBBBBBB
#define READVALIDMASK 0xCCCCCCCC
#define READINVALIDMASK 0xDDDDDDDD

class DirectShmem final
{
public:
    DirectShmem(float *dataPtr, int* shmemPtr) : data_(dataPtr), shmem_(shmemPtr) {};
    DirectShmem() = delete;

    float *getDataPtr() { return data_; };

    void setWriteMaskValid();

    void setWriteMaskInvalid();

    void setReadMaskValid();

    void setReadMaskInvalid();

    bool checkWriteMaskStatusValid();

    bool checkReadMaskStatusValid();

private:
    float *data_;
    int   *shmem_;

    bool isDataValid_ = false;
};