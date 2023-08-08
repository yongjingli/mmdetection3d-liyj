/*
 * Copyright (c) 2019, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#include <cuda_fp16.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include "NvInfer.h"
#include "ResizeBilinear.hpp"  // Resize for upsample
#include "UpsampleInt8.h"

template <typename Data>
__global__ void Upsample_kernel_2d(int nbatch, const float2 scale,
                                   const int2 osize, Data const *idata,
                                   const int istride, const int ibatchstride,
                                   Data *odata, const int ostride,
                                   const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;
  const int obstride = gridDim.z * obatchstride;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
    int iy = int(oy / scale.y);
    for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
      int ix = int(ox / scale.x);
      int o_offset = z0 * obatchstride + oy * ostride + ox;
      int i_offset = z0 * ibatchstride + iy * istride + ix;
      for (int batch = z0; batch < nbatch; batch += gridDim.z) {
        odata[o_offset] = idata[i_offset];
        i_offset += ibstride;
        o_offset += obstride;
      }
    }
  }
}

extern "C" int UpsampleV2Forward(int batchSize, float mSale, Dims &mInputDims,
                                 Dims &mOutputDims, DataType &mDataType,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    DPRINTF(1, "Skip RoIAlignPlugin::enqueue!!\n");
    return 0;
  }

  int nchan = mInputDims.d[0];
  float2 scale = {mSale, mSale};
  int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
  int istride = mInputDims.d[2];
  int ostride = mOutputDims.d[2];
  int ibatchstride = mInputDims.d[1] * istride;
  int obatchstride = mOutputDims.d[1] * ostride;
  dim3 block(32, 16);
  dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1,
            std::min(batchSize * nchan, 65535));
  if (mDataType == nvinfer1::DataType::kFLOAT) {
    Upsample_kernel_2d<<<grid, block, 0, stream>>>(
        batchSize * nchan, scale, osize, static_cast<float const *>(inputs[0]),
        istride, ibatchstride, static_cast<float *>(outputs[0]), ostride,
        obatchstride);
  }
  if (mDataType == DataType::kINT8) {
    Upsample_kernel_2d<<<grid, block, 0, stream>>>(
        batchSize * nchan, scale, osize, static_cast<char const *>(inputs[0]),
        istride, ibatchstride, static_cast<char *>(outputs[0]), ostride,
        obatchstride);
  } else {
    Upsample_kernel_2d<<<grid, block, 0, stream>>>(
        batchSize * nchan, scale, osize, static_cast<__half const *>(inputs[0]),
        istride, ibatchstride, static_cast<__half *>(outputs[0]), ostride,
        obatchstride);
  }
  return cudaGetLastError() != cudaSuccess;
}



#if NV_TENSORRT_MAJOR >= 6
class UpsamplePluginV2Creator : public IPluginCreator {
 public:
  const char *getPluginName() const override { return "UpsampleV2"; }

  const char *getPluginVersion() const override { return "2"; }

  const PluginFieldCollection *getFieldNames() override {
    return &mFieldCollection;
  }

  IPluginV2 *createPlugin(const char *name,
                          const PluginFieldCollection *fc) override {
    auto plugin = new UpsamplePluginV2(*fc);
    mFieldCollection = *fc;
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                               size_t serialLength) override {
    auto plugin = new UpsamplePluginV2(serialData, serialLength);
    mPluginName = name;
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace;
  std::string mPluginName;
  PluginFieldCollection mFieldCollection{0, nullptr};
};


//kernel for BatchPadConcat plugin
template <typename Tin, typename Tout>
__global__ void BatchPadConcat_kernel(int nbatch, const int2 osize, Tin const *idata,
                                   const int istride, const int ibatchstride,
                                   Tout *odata, const int ostride,
                                   const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;
  const int obstride = gridDim.z * obatchstride;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
	  for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
	    int o_offset = z0 * obatchstride + oy * ostride + ox;
	    int i_offset = z0 * ibatchstride + oy * istride + ox;
	    for (int batch = z0; batch < nbatch; batch += gridDim.z) {
	      odata[o_offset] = idata[i_offset];
	      i_offset += ibstride;
	      o_offset += obstride;
	    }
	  }
	}
}

template <typename Tin>
__global__ void BatchPadConcat_half(int nbatch, const int2 osize, Tin const *idata,
                                   const int istride, const int ibatchstride,
                                   __half *odata, const int ostride,
                                   const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;
  const int obstride = gridDim.z * obatchstride;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {  
	  for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
	    int o_offset = z0 * obatchstride + oy * ostride + ox;
	    int i_offset = z0 * ibatchstride + oy * istride + ox;
	    for (int batch = z0; batch < nbatch; batch += gridDim.z) {
	      odata[o_offset] = __short2half_rn( idata[i_offset]);
	      i_offset += ibstride;
	      o_offset += obstride;
	    }
	  }
	}
}

//Support: FLOAT, HALF, INT8->HALF, 
extern "C" int BatchPadConcatForward(int batchSize, Dims &mInputDims,
                                 Dims &mOutputDims, DataType &mInType, 
                                 DataType &mOutType, void **weights,
                                 const void *const *inputs, void **outputs,
                                 void *ws, cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    DPRINTF(1, "Skip BatchPadConcatForward!!\n");
    return 0;
  }

  int nchan = mInputDims.d[0];
  int ochan = mOutputDims.d[0];
  int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
  int istride = mInputDims.d[2];
  int ostride = mOutputDims.d[2];
  int ibatchstride = mInputDims.d[1] * istride;
  int obatchstride = mOutputDims.d[1] * ostride;
  dim3 block(32, 16);
  dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1,
            std::min(nchan, 65535));
   
  int iIdx = 0;  
  int eltSize = (mOutType == DataType::kHALF)? 2:4; 
  auto oBuf = outputs[0];      							  							         
  for( int nb = 0; nb<batchSize; nb++){
   if (mInType == mOutType) {
  	if (mOutType == DataType::kHALF) {
    	BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
        nchan, osize, static_cast<__half const *>(inputs[0])+iIdx,
        istride, ibatchstride, static_cast<__half *>(oBuf), ostride,
        obatchstride);
    }else if (mOutType == DataType::kFLOAT) {
    	BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
        nchan, osize, static_cast<float const *>(inputs[0])+iIdx,
        istride, ibatchstride, static_cast<float *>(oBuf), ostride,
        obatchstride);
    }
   } else if (mOutType == DataType::kHALF) {
    BatchPadConcat_half<<<grid, block, 0, stream>>>(
        nchan, osize, static_cast<char const *>(inputs[0])+iIdx,
        istride, ibatchstride, static_cast<__half *>(oBuf), ostride,
        obatchstride);
   } else {
    DPRINTF(0, "Unsupported! Skip BatchPadConcatForward!!\n");
   }
   cudaMemcpyAsync( oBuf + nchan * obatchstride * eltSize, 
   									weights[nb], (ochan-nchan) * obatchstride * eltSize,
                    cudaMemcpyDeviceToDevice, stream);
                                 
   iIdx += nchan * ibatchstride;
   oBuf = oBuf + ochan * obatchstride * eltSize; 
  }
  return cudaGetLastError() != cudaSuccess;
}

class BatchPadConcatPluginCreator : public IPluginCreator {
 public:
  const char *getPluginName() const override { return "BatchConcatPad"; }

  const char *getPluginVersion() const override { return "1"; }

  const PluginFieldCollection *getFieldNames() override {
    return &mFieldCollection;
  }

  IPluginV2 *createPlugin(const char *name,
                          const PluginFieldCollection *fc) override {
    auto plugin = new UpsamplePluginV2(*fc);
    mFieldCollection = *fc;
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                               size_t serialLength) override {
    auto plugin = new BatchPadConcatPlugin(serialData, serialLength);
    mPluginName = name;
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace;
  std::string mPluginName;
  PluginFieldCollection mFieldCollection{0, nullptr};
};

REGISTER_TENSORRT_PLUGIN(UpsamplePluginV2Creator);
REGISTER_TENSORRT_PLUGIN(BatchPadConcatPluginCreator);

#endif  // NV_TENSORRT_MAJOR >= 6