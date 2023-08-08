

#include "ResizeNearest.h"



template <typename Data>
__global__
void resize_nearest_kernel_2d(int nbatch,
                              float2 scale,
                              int2 osize,
                              Data const* idata, int istride, int ibatchstride,
                              Data*       odata, int ostride, int obatchstride) {
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

int ResizeNearestForward(int batchSize,const void *const *inputs, void *const*outputs,
                                 void *workspace, cudaStream_t stream,Dims input_dims,
								 Dims _output_dims,float _scale[],int _ndims,bool type_float) {
  //auto const& input_dims = this->getInputDims(0);
  int nchan = input_dims.d[0];
  switch( _ndims ) {
  case 2: {
    float2 scale = {_scale[1], _scale[0]};
    int2 osize = {_output_dims.d[2], _output_dims.d[1]};
    int istride =   input_dims.d[2];
    int ostride = _output_dims.d[2];
    int ibatchstride =   input_dims.d[1] * istride;
    int obatchstride = _output_dims.d[1] * ostride;
    dim3 block(32, 16);
    dim3 grid((osize.x - 1) / block.x + 1,
              (osize.y - 1) / block.y + 1,
              std::min(batchSize * nchan, 65535));
    if (type_float) {				
      resize_nearest_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, osize,
         static_cast<float const*>( inputs[0]), istride, ibatchstride,
         static_cast<float*      >(outputs[0]), ostride, obatchstride);
    } else {
      resize_nearest_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, osize,
         static_cast<__half const*>( inputs[0]), istride, ibatchstride,
         static_cast<__half*      >(outputs[0]), ostride, obatchstride);
    }
    return cudaGetLastError() != cudaSuccess;
  }
  
  }
  return 0;
}
