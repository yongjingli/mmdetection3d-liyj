#ifndef CUDA_KERNEL_H_
#define CUDA_KERNEL_H_
#include <cuda_runtime.h>
#include <cuda_fp16.h>
//int8 to fp16; chw to chw16;input:3*h*w;output H*W*16;
__global__ void CHW3Int8ToCHW16HalfKernel(const uint8_t* input,half* output,int width,int height)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    int one_chl=width*height;
    if(idx<3*one_chl){

        int c=idx/one_chl;        //0~2
        int h=(idx/width)%height;//0~height-1
        int w=idx%width;         //0~width-1

        int output_idx=h*width*16+w*16+c;
        output[output_idx]=__uint2half_rn(input[idx]);
   }
}
//for one channel 
__global__ void HWInt8ToCHW16HalfKernel(const uint8_t* input,half* output,int width,int height)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx<width*height){ 

        int h=idx/width; //0~height-1
        int w=idx%width; //0~width-1

        int output_idx=h*width*16+w*16;
        half data=__uint2half_rn(input[idx]);
        output[output_idx]=data;
        output[output_idx+1]=data;
        output[output_idx+2]=data;
   }
}

//for one channel 
__global__ void HWFp32ToCHW16HalfKernel(const uint8_t* input,half* output,int width,int height)
{
    int idx=threadIdx.x+blockIdx.x*blockDim.x;
    if(idx<width*height){ 

        int h=idx/width; //0~height-1
        int w=idx%width; //0~width-1

        int output_idx=h*width*16+w*16;
        half data=__float2half_rn(float(input[idx])/64.0f);
        output[output_idx]=data;
        output[output_idx+1]=data;
        output[output_idx+2]=data;
   }
}


#endif