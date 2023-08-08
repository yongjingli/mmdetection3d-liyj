#include "BatchConcatPad.h"

// Static class fields initialization
PluginFieldCollection BatchPadConcatPluginCreator::mFC{};
std::vector<PluginField> BatchPadConcatPluginCreator::mPluginAttributes;

// kernel for BatchPadConcat plugin
template <typename Tin, typename Tout>
__global__ void BatchPadConcat_kernel(int nbatch, const int2 isize, const int2 osize, Tin const *idata,
                                      const int istride, const int ibatchstride, Tout *odata, const int ostride,
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
      if (oy < isize.y && ox < isize.x) {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = idata[i_offset];
          i_offset += ibstride;
          o_offset += obstride;
        }
      } else {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = 0;
          o_offset += obstride;
        }
      }
    }
  }
}

template <typename Tin>
__global__ void BatchPadConcat_half(int nbatch, const int2 isize, const int2 osize, Tin const *idata, const int istride,
                                    const int ibatchstride, __half *odata, const int ostride, const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;
  const int obstride = gridDim.z * obatchstride;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
    for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
      int o_offset = z0 * obatchstride + oy * ostride + ox;
      int i_offset = z0 * ibatchstride + oy * istride + ox;
      if (oy < isize.y && ox < isize.x) {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = __short2half_rn(idata[i_offset]);
          i_offset += ibstride;
          o_offset += obstride;
        }
      } else {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = 0;
          o_offset += obstride;
        }
      }
    }
  }
}

template <typename Tin>
__global__ void BatchPadConcat_int8(int nbatch, const int2 isize, const int2 osize, Tin const *idata, const int istride,
                                    const int ibatchstride, char *odata, const int ostride, const int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;
  const int obstride = gridDim.z * obatchstride;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
    for (int ox = x0; ox < osize.x; ox += blockDim.x * gridDim.x) {
      int o_offset = z0 * obatchstride + oy * ostride + ox;
      int i_offset = z0 * ibatchstride + oy * istride + ox;
      if (oy < isize.y && ox < isize.x) {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = static_cast<char>(idata[i_offset]);
          i_offset += ibstride;
          o_offset += obstride;
        }
      } else {
        for (int batch = z0; batch < nbatch; batch += gridDim.z) {
          odata[o_offset] = 0;
          o_offset += obstride;
        }
      }
    }
  }
}
template <typename Tin, typename Tout>
__global__ void linear2chw32_packed_kernel(const int nchan, const int cpad, const int ochan, const int2 isize,
                                           const int2 osize, Tin const *idata, Tin const *weights, Tout *odata) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  for (int oy = y0; oy < osize.y; oy += blockDim.y * gridDim.y) {
    for (int ox = x0; ox < isize.x; ox += blockDim.x * gridDim.x) {
      my_int256_t i256;
      int t_offset = 0;
      int o_offset = oy * osize.x + ox;
      int i_offset = 0;
      if (oy < isize.y && ox < isize.y) {
        for (int ic = 0; ic < nchan; ic++) {
          i_offset = (ic * isize.y + oy) * isize.x + ox;
          i256.i8[t_offset++] = idata[i_offset];
        }
        for (int ic = 0; ic < cpad; ic++) {
          i_offset = (ic * osize.y + oy) * osize.x + ox;
          i256.i8[t_offset++] = weights[i_offset];
        }
        int reschan = ochan - nchan - cpad;
        for (int ic = 0; ic < reschan; ic++) {
          i256.i8[t_offset++] = 0;
        }
        odata[o_offset] = i256.i256;
      } else {
        // Maybe i256 is initialized with zero?
        for (int ic = 0; ic < ochan; ic ++){
          i256.i8[ic] = 0;
        }
        odata[o_offset] = i256.i256;
      }
    }
  }
}
template <typename Tin, typename Tout>
__global__ void linear2chwN_kernel(const int nchan, const int aligment, const int start_cn, const int2 isize,
                                   const int2 osize, Tin const *idata, const int istride, const int ibatchstride,
                                   Tout *odata) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  const int ibstride = gridDim.z * ibatchstride;

  for (int iy = y0; iy < isize.y; iy += blockDim.y * gridDim.y) {
    for (int ix = x0; ix < isize.x; ix += blockDim.x * gridDim.x) {
      int i_offset = z0 * ibatchstride + iy * istride + ix;
      int r_z0 = z0 + start_cn;
      int o_offset = ((r_z0 / aligment * osize.y + iy) * osize.x + ix) * aligment + r_z0 % aligment;
      for (int ic = z0; ic < nchan; ic += gridDim.z) {
        odata[o_offset] = idata[i_offset];
        i_offset += ibstride;
      }
    }
  }
}

int BatchPadConcatForward(int batchSize, Dims &mInputDims, Dims &mOutputDims, DataType &mInType,
                                     DataType &mOutType, void **weights, const void *const *inputs, void *const*outputs,
                                     void *ws, cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    DPRINTF(1, "Skip BatchPadConcatForward!!\n");
    return 0;
  }
  cudaDebugError(2);
  
  int nchan = mInputDims.d[0];
  int ochan = mOutputDims.d[0];
  int2 isize = {mInputDims.d[2], mInputDims.d[1]};
  int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
  int istride = mInputDims.d[2];
  int ostride = mOutputDims.d[2];
  int ibatchstride = mInputDims.d[1] * istride;
  int obatchstride = mOutputDims.d[1] * ostride;
  dim3 block(32, 16);
  dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(nchan, 65535));

  int iIdx = 0;
  int eltSize = elementSize(mOutType);
  // int eltSize = (mOutType == DataType::kHALF)? 2:4;
  auto oBuf = outputs[0];
  for (int nb = 0; nb < batchSize; nb++) {
    if (mInType == mOutType) {
      if (mOutType == DataType::kHALF) {
        BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
            nchan, isize, osize, static_cast<__half const *>(inputs[0]) + iIdx, istride, ibatchstride,
            static_cast<__half *>(oBuf), ostride, obatchstride);
      } else if (mOutType == DataType::kFLOAT) {
        BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
            nchan, isize, osize, static_cast<float const *>(inputs[0]) + iIdx, istride, ibatchstride,
            static_cast<float *>(oBuf), ostride, obatchstride);
      } else if (mOutType == DataType::kINT8) {
        BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
            nchan, isize, osize, static_cast<char const *>(inputs[0]) + iIdx, istride, ibatchstride,
            static_cast<char *>(oBuf), ostride, obatchstride);
      }
    } else if (mOutType == DataType::kINT8) {
      if (mInType == DataType::kFLOAT) {
        BatchPadConcat_int8<<<grid, block, 0, stream>>>(nchan, isize, osize,
                                                        static_cast<float const *>(inputs[0]) + iIdx, istride,
                                                        ibatchstride, static_cast<char *>(oBuf), ostride, obatchstride);
      }
    } else if (mOutType == DataType::kHALF) {
      BatchPadConcat_half<<<grid, block, 0, stream>>>(nchan, isize, osize, static_cast<char const *>(inputs[0]) + iIdx,
                                                      istride, ibatchstride, static_cast<__half *>(oBuf), ostride,
                                                      obatchstride);
    } else {
      DPRINTF(0, "Unsupported! Skip BatchPadConcatForward!!\n");
    }
    cudaMemcpyAsync((char *)oBuf + nchan * obatchstride * eltSize, weights[nb],
                    (ochan - nchan) * obatchstride * eltSize, cudaMemcpyDeviceToDevice, stream);

    iIdx += nchan * ibatchstride;
    oBuf = (char *)oBuf + ochan * obatchstride * eltSize;
  }
  
  cudaDebugError(2);
  return cudaGetLastError() != cudaSuccess;
}
int BatchPadConcatForwardV2(int batchSize, Dims &inDims, Dims outDims, DataType inType, DataType outType,
                                       TensorFormat inFormat, TensorFormat outFormat, int cPad, void **weights,
                                       const void *const *inputs, void *const*outputs, void *workspace,
                                       cudaStream_t stream) {
  int nchan = inDims.d[0];
  int ochan = outDims.d[0];
  int2 isize = {inDims.d[2], inDims.d[1]};
  int2 osize = {outDims.d[2], outDims.d[1]};
  int istride = inDims.d[2];
  int ostride = outDims.d[2];
  int ibatchstride = inDims.d[1] * istride;
  int obatchstride = outDims.d[1] * ostride;
  int alignment = 1;
  if (TensorFormat::kCHW4 == outFormat) {
    alignment = 4;
  } else if (TensorFormat::kCHW32 == outFormat) {
    alignment = 32;
  }
  int outEltSize = elementSize(outType);
  int ibatch_offset = 0;
  int obatch_offset = 0;
  dim3 block(32, 16);

  for (int nb = 0; nb < batchSize; nb++) {
    if (inType == outType) {
      if (DataType::kINT8 == outType) {
        if (TensorFormat::kCHW32 == outFormat && TensorFormat::kLINEAR == inFormat && ochan <= 32) {
          dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, 1);
          linear2chw32_packed_kernel<<<grid, block, 0, stream>>>(
            nchan, cPad, ochan, isize, osize, static_cast<int8_t const *>(inputs[0]) + ibatch_offset,
              static_cast<int8_t const *>(weights[nb]), static_cast<longlong4 *>(outputs[0]));          
        } else {
          dim3 grid_images((isize.x - 1) / block.x + 1, (isize.y - 1) / block.y + 1, std::min(nchan, 65535));
          dim3 grid_weights((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(cPad, 65535));
          cudaMemsetAsync(static_cast<char *>(outputs[0]) + obatch_offset * outEltSize, 0, ochan * obatchstride * outEltSize,
                stream);
          linear2chwN_kernel<<<grid_images, block, 0, stream>>>(
              nchan, alignment, 0, isize, osize, static_cast<int8_t const *>(inputs[0]) + ibatch_offset, istride,
              ibatchstride, static_cast<int8_t *>(outputs[0]) + obatch_offset);
          linear2chwN_kernel<<<grid_weights, block, 0, stream>>>(
              cPad, alignment, nchan, isize, osize, static_cast<int8_t const *>(weights[nb]), ostride, obatchstride,
              static_cast<int8_t *>(outputs[0]) + obatch_offset);
        }
      } else if (DataType::kFLOAT == outType) {
        dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(cPad, 65535));
        BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
            nchan, isize, osize, static_cast<float const *>(inputs[0]) + ibatch_offset, istride, ibatchstride,
            static_cast<float *>(outputs[0]) + obatch_offset, ostride, obatchstride);
        cudaMemcpyAsync(static_cast<float *>(outputs[0]) + obatch_offset + nchan * obatchstride, weights[nb],
                        cPad * obatchstride * outEltSize, cudaMemcpyDeviceToDevice, stream);
      } else if (DataType::kHALF == outType) {
        dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(cPad, 65535));
        BatchPadConcat_kernel<<<grid, block, 0, stream>>>(
            nchan, isize, osize, static_cast<__half const *>(inputs[0]) + ibatch_offset, istride, ibatchstride,
            static_cast<__half *>(outputs[0]) + obatch_offset, ostride, obatchstride);
        cudaMemcpyAsync(static_cast<__half *>(outputs[0]) + obatch_offset + nchan * obatchstride, weights[nb],
                        cPad * obatchstride * outEltSize, cudaMemcpyDeviceToDevice, stream);
      }
    }
    ibatch_offset += nchan * ibatchstride;
    obatch_offset += ochan * obatchstride;
  }
  return cudaGetLastError() != cudaSuccess;
}