/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CUDANVSCI_H
#define CUDANVSCI_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nvscibuf.h>
#include <nvscisync.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <vector>

#define checkNvSciErrors(call)                              \
  do {                                                      \
    NvSciError _status = call;                              \
    if (NvSciError_Success != _status) {                    \
      printf(                                               \
          "NVSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, NvSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

class CudaNvSciSignal {
 private:
  NvSciSyncModule m_syncModule;
  NvSciBufModule m_bufModule;

  NvSciSyncAttrList m_syncAttrList;
  NvSciSyncFence *m_fence;

  NvSciBufAttrList m_tensorBufAttrList;
  // NvSciBufAttrList m_imageBufAttrList;
  NvSciBufAttrList m_buffAttrListOut[2];
  NvSciBufAttrKeyValuePair pairArrayOut[10];

  cudaExternalMemory_t extMemRawBuf, extMemImageBuf;
  // cudaMipmappedArray_t d_mipmapArray;
  // cudaArray_t d_mipLevelArray;
  // cudaTextureObject_t texObject;
  cudaExternalSemaphore_t signalSem;

  int m_cudaDeviceId;
  cudaStream_t streamToRun;
  // CUuuid m_devUUID;
  // uint64_t m_imageWidth;
  // uint64_t m_imageHeight;
  void *d_outputBuf;
  // size_t m_bufSize;

 public:
  CudaNvSciSignal(NvSciBufModule bufModule, NvSciSyncModule syncModule, int cudaDeviceId, NvSciSyncFence *fence,
                  cudaStream_t stream)
      : m_syncModule(syncModule),
        m_bufModule(bufModule),
        m_fence(fence),
        m_cudaDeviceId(cudaDeviceId),
        streamToRun(stream),
        d_outputBuf(NULL) {
    initCuda();

    checkNvSciErrors(NvSciSyncAttrListCreate(m_syncModule, &m_syncAttrList));
    checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_tensorBufAttrList));
    // checkNvSciErrors(NvSciBufAttrListCreate(m_bufModule, &m_imageBufAttrList));

    setTensorBufAttrList();
    // setImageBufAttrList(m_imageWidth, m_imageHeight);

    checkCudaErrors(cudaDeviceGetNvSciSyncAttributes(m_syncAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal));
  }

  ~CudaNvSciSignal() {
    // checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    // checkCudaErrors(cudaFreeMipmappedArray(d_mipmapArray));
    checkCudaErrors(cudaFree(d_outputBuf));
    checkCudaErrors(cudaDestroyExternalSemaphore(signalSem));
    checkCudaErrors(cudaDestroyExternalMemory(extMemRawBuf));
    // checkCudaErrors(cudaDestroyExternalMemory(extMemImageBuf));
    // checkCudaErrors(cudaDestroyTextureObject(texObject));
    // checkCudaErrors(cudaStreamDestroy(streamToRun));
  }

  void initCuda() {
    // checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    // checkCudaErrors(
    //     cudaStreamCreateWithFlags(&streamToRun, cudaStreamNonBlocking));

    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, m_cudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, m_cudaDeviceId));
    printf(
        "[CudaNvSciSignal] GPU Device %d: \"%s\" with compute capability "
        "%d.%d\n\n",
        m_cudaDeviceId, _ConvertSMVer2ArchName(major, minor), major, minor);

    // CUresult res = cuDeviceGetUuid(&m_devUUID, m_cudaDeviceId);
    // if (res != CUDA_SUCCESS) {
    //   fprintf(stderr, "Driver API error = %04d \n", res);
    //   exit(EXIT_FAILURE);
    // }
  }

  void setTensorBufAttrList() {
    NvSciBufType bufType = NvSciBufType_Tensor;  // NvSciBufType_RawBuffer;
    bool cpuAccess = false;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair rawBufAttrs[] = {
        // {NvSciBufRawBufferAttrKey_Size, &size, sizeof(size)},
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        // {NvSciBufGeneralAttrKey_GpuId, &m_devUUID, sizeof(m_devUUID)},
    };

    checkNvSciErrors(NvSciBufAttrListSetAttrs(m_tensorBufAttrList, rawBufAttrs,
                                              sizeof(rawBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
  }

  NvSciSyncAttrList getNvSciSyncAttrList() { return m_syncAttrList; }

  NvSciBufAttrList getNvSciRawBufAttrList() { return m_tensorBufAttrList; }

  void cudaImportNvSciSemaphore(NvSciSyncObj syncObj) {
    // checkCudaErrors(cudaSetDevice(m_cudaDeviceId));

    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = (void *)syncObj;

    checkCudaErrors(cudaImportExternalSemaphore(&signalSem, &extSemDesc));
  }

  void signalExternalSemaphore() {
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    // For cross-process signaler-waiter applications need to use NvSciIpc
    // and NvSciSync[Export|Import] utilities to share the NvSciSyncFence
    // across process. This step is optional in single-process.
    signalParams.params.nvSciSync.fence = (void *)m_fence;
    signalParams.flags = 0;

    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&signalSem, &signalParams, 1, streamToRun));
  }

  void *cudaImportNvSciTensorBuf(NvSciBufObj inputBufObj) {
    // checkCudaErrors(cudaSetDevice(m_cudaDeviceId));
    checkNvSciErrors(NvSciBufObjGetAttrList(inputBufObj, &m_buffAttrListOut[0]));

    memset(pairArrayOut, 0, sizeof(NvSciBufAttrKeyValuePair) * 10);
    pairArrayOut[0].key = NvSciBufTensorAttrKey_DataType;
    pairArrayOut[1].key = NvSciBufTensorAttrKey_NumDims;
    pairArrayOut[2].key = NvSciBufTensorAttrKey_SizePerDim;
    pairArrayOut[3].key = NvSciBufTensorAttrKey_AlignmentPerDim;
    pairArrayOut[4].key = NvSciBufTensorAttrKey_StridesPerDim;

    checkNvSciErrors(NvSciBufAttrListGetAttrs(m_buffAttrListOut[0], pairArrayOut, 5));

    // int32_t num_dim = *(int32_t *)pairArrayOut[1].value;
    // NvSciBufAttrValDataType type = *(NvSciBufAttrValDataType*)pairArrayOut[0].value;
    uint64_t *size_per_dim = (uint64_t *)pairArrayOut[2].value;
    // uint32_t* alignment_per_dim = (uint32_t *)pairArrayOut[3].value;
    uint64_t *strides_per_dim = (uint64_t *)pairArrayOut[4].value;
    /*
    std::cout << "num_dim: " << num_dim << ", type: " << type << ", size_per_dim: (" <<
      size_per_dim[0] << ", " << size_per_dim[1] << ", " << size_per_dim[2] << ", " << size_per_dim[3] << ", " <<
    size_per_dim[4] << "), alignment_per_dim: (" << alignment_per_dim[0] << ", " << alignment_per_dim[1] << ", " <<
    alignment_per_dim[2] << ", " << alignment_per_dim[3] << ", " << alignment_per_dim[4] << "), strides_per_dim: (" <<
      strides_per_dim[0] << ", " << strides_per_dim[1] << ", " << strides_per_dim[2] << ", " << strides_per_dim[3] << ",
    " << strides_per_dim[4] << ")" << std::endl;
    */
    unsigned long long max_size = size_per_dim[0] * strides_per_dim[0];

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    // memHandleDesc.handle.nvSciBufObject = inputBufObj;
    printf("comment out nvSciBufObject.\n");
    memHandleDesc.size = 128;
    checkCudaErrors(cudaImportExternalMemory(&extMemRawBuf, &memHandleDesc));

    cudaExternalMemoryBufferDesc bufferDesc;
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = max_size;

    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&d_outputBuf, extMemRawBuf, &bufferDesc));

    return d_outputBuf;
  }
};

#endif