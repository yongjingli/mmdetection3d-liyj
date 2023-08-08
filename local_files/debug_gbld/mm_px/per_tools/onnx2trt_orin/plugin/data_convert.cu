#include <cuda.h>
#include <cuda_fp16.h>
#include "data_convert.h"
#include "ResizeBilinear.hpp"

#define THREAD_X 1024
#define THREAD_Y 512
#define THREAD_Z 256
#define diveUp(n, t) ((n - 1) / t + 1)
__global__ void res2channel_kernel_int8(const uint8_t* src, int8_t* dst,
                                                const uint32_t width,
                                                const uint32_t height, 
                                                const uint32_t data_num) {
    const uint8_t *batch_src = src + blockIdx.y * data_num;
    int8_t *batch_dst = dst + blockIdx.y * data_num;                                                  
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < data_num; i += stride) {
        const int c = i / (width * height);
        const int y = (i % (width * height)) / width;
        const int x = (i % (width * height)) % width;

        int dst_c = ((x & 1) * 2 + (y & 1)) * 3 + c;
        int dst_y = y / 2;
        int dst_x = x / 2;
        int dst_idx = dst_c * (width / 2) * (height / 2) + dst_y * (width / 2) + dst_x;    

        batch_dst[dst_idx] = (int8_t)(batch_src[idx] / 2);
    }
}

void Res2ChannelWrapperInt8(const char* inDataPtr, char* outPtr, int dataNum,
                            int batchSize, int ch, int width, int height, cudaStream_t stream) {
    dim3 GridSize(8, batchSize, 1);
    dim3 BlockSize(THREAD_X, 1, 1);

    res2channel_kernel_int8<<<GridSize, BlockSize, 0, stream>>>
                                ((const uint8_t*) inDataPtr, (int8_t*) outPtr, width, height, dataNum);
}

template<typename T>
__global__ void res2channel_kernel_float32(const T* src, float* dst,
                                            const uint32_t width,
                                            const uint32_t height,
                                            const uint32_t data_num) {
    const T *batch_src = src + blockIdx.y * data_num;
    float *batch_dst = dst + blockIdx.y * data_num;                                            
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = blockDim.x * gridDim.x; // 256*8
    for(int i = idx; i < data_num; i += stride) {
        const int c = i / (width * height);
        const int y = (i % (width * height)) / width;
        const int x = (i % (width * height)) % width;

        int dst_c = ((x & 1) * 2 + (y & 1)) * 3 + c;
        int dst_y = y / 2;
        int dst_x = x / 2;
        int dst_idx = dst_c * (width / 2) * (height / 2) + dst_y * (width / 2) + dst_x;    

        batch_dst[dst_idx] = (float)batch_src[i] / 255.0f;
    }
}

template<typename T>
void Res2ChannelWrapperFloat32(const T* inDataPtr, char* outPtr, int dataNum,
                                int batchSize, int ch, int width, int height, cudaStream_t stream) {
    dim3 GridSize(8, batchSize, 1);
    dim3 BlockSize(THREAD_X, 1, 1);

    res2channel_kernel_float32<<<GridSize, BlockSize, 0, stream>>>
                                (inDataPtr, (float*)outPtr, width, height, dataNum);
} 

__global__ void convert_kernel_int8(const uint8_t* src, char* dst,
                                            const uint32_t data_num) {
    const uint8_t* batch_src = src + blockIdx.y * data_num;
    char *batch_dst = dst + blockIdx.y * data_num;                                            
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < data_num; i += stride) {
        batch_dst[i] = int(batch_src[i] / 2);
    }
}

void ConvertWrapperInt8(const char* inDataPtr, char* outPtr, int dataNum,
                                int batchSize, cudaStream_t stream) {
    dim3 GridSize(8, batchSize, 1);
    dim3 BlockSize(THREAD_X, 1, 1);

    convert_kernel_int8<<<GridSize, BlockSize, 0, stream>>>
                                ((const uint8_t*)inDataPtr, outPtr, dataNum);
}

template<typename T>
__global__ void convert_kernel_float32(const T* src, float* dst, const uint32_t data_num) {
    const T *batch_src = src + blockIdx.y * data_num;
    float *batch_dst = dst + blockIdx.y * data_num;
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < data_num; i += stride) {
        batch_dst[i] = (float)batch_src[i] / 255.0f;
    }   
}

template<typename T>
void ConvertWrapperFloat32(const T* inDataPtr, char* outPtr, int batchSize, int dataNum, cudaStream_t stream) {
    dim3 GridSize(8, batchSize, 1);
    dim3 BlockSize(THREAD_X, 1, 1);

    convert_kernel_float32<<<GridSize, BlockSize, 0, stream>>>
                                (inDataPtr, (float*) outPtr, dataNum);
} 

void TransFormat(char const* inDataPtr, int batchSize, int ch, int inDataType,
                 EngineBuffer *bufferInfo, cudaStream_t stream){
    int dstDataType = bufferInfo[0].nDataType;
    int dstCH = bufferInfo[0].d[0];
    int height = bufferInfo[0].d[1];
    int width  = bufferInfo[0].d[2];
    int dataNum = 1;
    for(int iter = 0; iter < 3 && bufferInfo[0].d[iter] > 0; ++iter) {
      dataNum *= bufferInfo[0].d[iter];
    }
    char* outPtr = (char*)bufferInfo[0].p;
    
    DPRINTF(2, "The height:%d; width:%d  dataNum:%d inDataType:%d\n", height, width, dataNum, inDataType);
    if (12 == dstCH) {
        if (2 == inDataType) { // 2: INT8
          if (2 == dstDataType) { 
            // UINT8 3HW -> INT8 12HW (/2)
            DPRINTF(2, "Res2ChannelWrapperInt8: UINT8_3HW -> INT8_12HW (/2) %d\n", fflush(stdout));
            Res2ChannelWrapperInt8(inDataPtr, outPtr, dataNum, batchSize, 3, width * 2, height * 2, stream);
          } else { 
            // UINT8 3HW -> FP32 12HW (/255)
            DPRINTF(2, "Res2ChannelWrapperFloat32: UINT8_3HW -> FP32_12HW (/255) %d\n", fflush(stdout));
            Res2ChannelWrapperFloat32((const uint8_t*)inDataPtr, outPtr, dataNum, batchSize, 3, width * 2, height * 2, stream);
          }
        } else { // 0: FP32 3HW -> FP32 12HW (/255)
          DPRINTF(2, "Res2ChannelWrapperFloat32: FP32_3HW -> FP32_12HW (/255\n) %d\n", fflush(stdout));
          Res2ChannelWrapperFloat32((const float*) inDataPtr, outPtr, dataNum, batchSize, 3, width * 2, height * 2, stream);
        }
    } else if ( 3 == dstCH ) {
        if( 2 == inDataType ) { // 2: INT8
          if ( 2 == dstDataType ) { 
            // UINT8 3HW -> INT8 3HW (/2)
            DPRINTF(2, "ConvertWrapperInt8: UINT8_3HW -> INT8_3HW (/2) %d\n", fflush(stdout));
            ConvertWrapperInt8(inDataPtr, outPtr, dataNum, batchSize, stream);
          } else { 
            // UINT8 3HW -> FP32 3HW (/255)
            DPRINTF(2, "ConvertWrapperInt8: UINT8_3HW -> FP32_3HW (/255) %d\n", fflush(stdout));
            ConvertWrapperFloat32((const uint8_t*)inDataPtr, outPtr, dataNum, batchSize, stream);
          }          
        } else { // 0: FP32 3HW -> FP32 3HW  (/255)
          DPRINTF(2, "ConvertWrapperInt8: FP32_3HW -> FP32_3HW (/255) %d\n", fflush(stdout));      
          ConvertWrapperFloat32((const float*) inDataPtr, outPtr, dataNum, batchSize, stream);
        }   
    } else {
      // Not support yet
      DPRINTF(1, "TransFormat: Not support yet! %d\n", fflush(stdout));
    }
}
                 

