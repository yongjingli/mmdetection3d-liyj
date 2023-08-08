/*
 * Copyright (c) 2019, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <iostream>

#include "NvInfer.h"
#include "UpsampleInt8.h"

// Static class fields initialization
PluginFieldCollection UpsamplePluginV2Creator::mFC{};
std::vector<PluginField> UpsamplePluginV2Creator::mPluginAttributes;

template <DataType in, DataType out>
void transform(const void *src, void *dst, int count)
{
  assert(in == out);
  memcpy(dst, src, count * elementSize(in));
}

template <>
void transform<DataType::kINT8, DataType::kFLOAT>(const void *src, void *dst, int count)
{
  auto srcPtr = static_cast<const int8_t *>(src);
  auto dstPtr = static_cast<float *>(dst);
  std::transform(srcPtr, srcPtr + count, dstPtr, [](int8_t in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kINT8>(const void *src, void *dst, int count)
{
  auto srcPtr = static_cast<const float *>(src);
  auto dstPtr = static_cast<int8_t *>(dst);
  std::transform(srcPtr, srcPtr + count, dstPtr, [](float x) {
    x = std::max(x, float(INT8_MIN));
    x = std::min(x, float(INT8_MAX));
    return static_cast<int8_t>(x);
  });
}

template <typename Data>
__global__ void Upsample_kernel_2dv2(int nbatch, const float2 scale, const int2 osize, Data const *idata,
                                   const int istride, const int ibatchstride, Data *odata, const int ostride,
                                   const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) {
        int ix = int(ox / scale.x);
        int iy = int(oy / scale.y);
        odata[batch * obatchstride + oy * ostride + ox] =
          idata[batch * ibatchstride + iy * istride + ix];
      }
    }
  }
}



int UpsampleV2ForwardV2(int batchSize, float mWScale, float mHSale, Dims &mInputDims, Dims &mOutputDims, DataType &mDataType,
                                   TensorFormat inTensorFormat, TensorFormat outTensorFormat, const void *const *inputs,
                                   void *const*outputs, void *workspace, cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    DPRINTF(1, "Skip RoIAlignPlugin::enqueue!!\n");
    return 0;
  }
  
  int nchan = mInputDims.d[0];
  float2 scale = {mWScale, mHSale};
  int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
  int istride = mInputDims.d[2];
  int ostride = mOutputDims.d[2];
  int ibatchstride = mInputDims.d[1] * istride;
  int obatchstride = mOutputDims.d[1] * ostride;
  dim3 block(32, 16);
  dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batchSize * nchan, 65535));
  if (mDataType == nvinfer1::DataType::kFLOAT) {
    Upsample_kernel_2dv2<<<grid, block, 0, stream>>>(batchSize * nchan, scale, osize,
                                                   static_cast<float const *>(inputs[0]), istride, ibatchstride,
                                                   static_cast<float *>(outputs[0]), ostride, obatchstride);
  } else if (mDataType == DataType::kINT8) {
    if(TensorFormat::kCHW32 == inTensorFormat && TensorFormat::kCHW32 == outTensorFormat){
      int alignedChan = (nchan + 31) / 32;
      dim3 grid32((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batchSize * alignedChan, 65535));      
      Upsample_kernel_2dv2<<<grid32, block, 0, stream>>>(batchSize * alignedChan, scale, osize,
                                                   static_cast<longlong4 const *>(inputs[0]), istride, ibatchstride,
                                                   static_cast<longlong4 *>(outputs[0]), ostride, obatchstride);
    } else {
      Upsample_kernel_2dv2<<<grid, block, 0, stream>>>(batchSize * nchan, scale, osize,
                                                    static_cast<char const *>(inputs[0]), istride, ibatchstride,
                                                    static_cast<char *>(outputs[0]), ostride, obatchstride);
    }
  } else {
    Upsample_kernel_2dv2<<<grid, block, 0, stream>>>(batchSize * nchan, scale, osize,
                                                   static_cast<__half const *>(inputs[0]), istride, ibatchstride,
                                                   static_cast<__half *>(outputs[0]), ostride, obatchstride);
  }

  return cudaGetLastError() != cudaSuccess;
}

