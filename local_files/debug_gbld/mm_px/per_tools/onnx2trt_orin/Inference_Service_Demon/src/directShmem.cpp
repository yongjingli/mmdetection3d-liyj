#include "directShmem.h"

void DirectShmem::setWriteMaskValid() {
    shmem_[0] = WRITEVALIDMASK;
}

void DirectShmem::setWriteMaskInvalid() {
    shmem_[0] = WRITEINVALIDMASK;
}

void DirectShmem::setReadMaskValid() {
    shmem_[0] = READVALIDMASK;
}

void DirectShmem::setReadMaskInvalid() {
    shmem_[0] = READINVALIDMASK;
}

bool DirectShmem::checkWriteMaskStatusValid() {
    if(shmem_[0] == WRITEVALIDMASK)   isDataValid_ = true;
    else                              isDataValid_ = false;

    return isDataValid_;
}

bool DirectShmem::checkReadMaskStatusValid() {
    if(shmem_[0] == READVALIDMASK)   isDataValid_ = true;
    else                             isDataValid_ = false;

    return isDataValid_;
}