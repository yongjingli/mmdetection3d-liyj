
#include "common.h"
// CUDA: use 256 threads per block
const int CAFFE_CUDA_NUM_THREADS = 256;
inline int CAFFE_GET_BLOCKS(const int N) { return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS; }
__global__ void float22Half2Vec(float2 *src, half2 *des, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < size / 2; i += stride) des[i] = __float22half2_rn(src[i]);
}
__global__ void half22Float2Vec(half2 *src, float2 *des, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < size / 2; i += stride) des[i] = __half22float2(src[i]);
}

__global__ void float2int8Vec(float *src, int8_t *des, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < size; i += stride) {
    des[i] = (int8_t)src[i];
  }
}
/*
float *pCPU : pointer to CPU data, float32
float *pCPU : pointer to GPU data, float32/float16
int size: the size of data
int type: 0: float32, 1: float16 2: int8
int direction: 0: CPU to GPU, 1: GPU to CPU
*/
int convertCPU_GPU(void *pCPU, void *pGPU, int size, int type, int direction, cudaStream_t stream, void *pBuffer) {
    if (size <= 0) return 0;
  
    if (0 == type) {
      if (0 == direction) {
        CHECK_CUDA(cudaMemcpyAsync(pGPU, pCPU, size * sizeof(float), cudaMemcpyHostToDevice, stream));
      } else {
        CHECK_CUDA(cudaMemcpyAsync(pCPU, pGPU, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
      }
    } else if (1 == type) {
      if (0 == direction) {
        CHECK_CUDA(cudaMemcpyAsync(pBuffer, pCPU, size * sizeof(float), cudaMemcpyHostToDevice, stream));
        float22Half2Vec<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS, 0, stream>>>((float2 *)pBuffer, (half2 *)pGPU,
                                                                                       size);
      } else {
        half22Float2Vec<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS, 0, stream>>>((half2 *)pGPU, (float2 *)pBuffer,
                                                                                       size);
        CHECK_CUDA(cudaMemcpyAsync(pCPU, pBuffer, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
      }
    } else if (2 == type) {
      if (0 == direction) {
        CHECK_CUDA(cudaMemcpyAsync(pBuffer, pCPU, size * sizeof(float), cudaMemcpyHostToDevice, stream));
        float2int8Vec<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS, 0, stream>>>((float *)pBuffer, (int8_t *)pGPU, size);
      } else {
      }
    }
  
    if (1 == direction) {
      cudaStreamSynchronize(stream);
    }
  
    return cudaGetLastError() != cudaSuccess;
  }
  