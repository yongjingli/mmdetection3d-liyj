/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12 for retinaNet
 *
 * Input: class(5),box(5)
 * input -> Decode -> Concat -> NMS -> output
 */
#include <string.h>
#include "ConfidenceFilterDLA.h"

#define DEBUG_GPU_PRINT() //{if(m<4096 && 2>(m%16)||(m>=69312&&m<69312+16))printf("m:%d i:%d conf:%f\n", m, index[0], conf);}

//For DLA fp16-chw16 and int8-chw32 , similar to hwc if c<=16
__device__ float getVal(__half val, float scale) { return __half2float(val); }
__device__ float getVal(signed char val, float scale) { return (val * scale); }

template <typename T, int CA, int FILTER_SUM_NUM>
__global__ void filter_FpnSum_DLA(const int num_detections, const int4 *input_dims, const int max_num, 
                           const int conf_offset, const float threshold,
                           const T **input, float *scale, const int col_in, float *output, 
                           const int fpn_w, int *count,int*input_size,int output_size) {
  __shared__ int4 input_dim_s[FILTER_SUM_NUM];
  __shared__ const T* conf_ptr_s[FILTER_SUM_NUM];
  if(threadIdx.x < FILTER_SUM_NUM){
    input_dim_s[threadIdx.x] = input_dims[threadIdx.x];
    conf_ptr_s[threadIdx.x] = input[threadIdx.x]+blockIdx.y*input_size[threadIdx.x];;
  }
  __syncthreads();

  // Go through detections by descending score
  const int step = blockDim.x * gridDim.x; // 256 * 8  or 256 * 1
  const int col_out = (col_in+1); // 3
  for (int m = blockDim.x * blockIdx.x + threadIdx.x; m < num_detections ; m+=step) {
    const int x_o = m % fpn_w;
    const int y_o = m / fpn_w;
    int index[FILTER_SUM_NUM];
    float conf = 0.f;
    for (int i = 0; i < FILTER_SUM_NUM; i++) {
      const int x_i = x_o / input_dim_s[i].z;
      const int y_i = y_o / input_dim_s[i].z;
      const int ch_i = x_o % input_dim_s[i].z + (y_o % input_dim_s[i].z) * input_dim_s[i].z; // + offset * input_dim_s[i].z * input_dim_s[i].z
      index[i] = (((ch_i / CA) * input_dim_s[i].y + y_i) * input_dim_s[i].x + x_i) * CA + (ch_i % CA);
      conf += getVal(conf_ptr_s[i][index[i]], scale[i]); 
    }
    DEBUG_GPU_PRINT();
    if( conf > threshold ) { // && count < max_in_block 
      int idx = atomicAdd( count+blockIdx.y, 1 );
      if( idx >= max_num ) continue;

      float *out_ptr = (output+blockIdx.y*output_size) + idx*col_out; 
      ((int*)out_ptr)[0] = m;
      out_ptr[1] = conf;        
      for (int k = 1; k < col_in; k++){
        if( 1 == FILTER_SUM_NUM ){      
          out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index[0]], scale[0]);
        } else {
          out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index[0]], scale[0]) + 
                         getVal(conf_ptr_s[1][input_dim_s[1].w * k + index[1]], scale[1]) + 
                         getVal(conf_ptr_s[2][input_dim_s[2].w * k + index[2]], scale[2]);
        }                                                    
      } 
      //printf("f%d m:%d conf:%f value:%f\n",blockIdx.y, idx, m, conf, out_ptr[2]);
    } 
  }
  //if(blockIdx.x==0 && threadIdx.x==0 )
  //printf("filter_FpnSum_kernel done\n");
}

//For DLA fp16 chw16, similar to hwc if c<=16
extern "C" int ConfidenceFilterForDLA(int batchSize, FilterParam *param, cudaStream_t stream) {
  const int max_blocks = 8;
  const int max_threads = 256;
  const int col_num = param->fpn_shape[0];                      
  float* output = (float*)param->outputs[0];
  const int output_size = param->max_num*(col_num+1);
  float &_conf_thresh = param->thresholds[0];
  int _fpn_mode = param->mode;
  float threshold =  (_fpn_mode & 0x01) ? logf(_conf_thresh/(1-_conf_thresh)) : _conf_thresh;
  
  cudaMemsetAsync(param->outputs[0], 0x0CC, batchSize*output_size*sizeof(float), stream);
  cudaMemsetAsync(param->_count, 0x0, batchSize*param->input_num*sizeof(int), stream);      
  { //FPNSum, for tfl2 header
    const int num_detections = param->fpn_shape[1] * param->fpn_shape[2];
    dim3 grid;
    grid.x=max_blocks;
    grid.y=batchSize;
    grid.z=1;
    if( 1 == param->data_type ){ // data_type = 1 fp16
      filter_FpnSum_DLA<__half, 16, 1><<<grid, max_threads, 0, stream>>>(
        num_detections, (const int4*)param->_input_dims, param->max_num, 
        param->_conf_offset[0], threshold, (const __half**)param->_d_input_bufs, param->out_scale, col_num, output, 
        param->fpn_shape[2], param->_count, param->_input_size, output_size);
    } else { // data_type = 2 int8
      filter_FpnSum_DLA<signed char, 32, 1><<<grid, max_threads, 0, stream>>>(
        num_detections, (const int4*)param->_input_dims, param->max_num, 
        param->_conf_offset[0], threshold, (const signed char**)param->_d_input_bufs, param->out_scale, col_num, output,
        param->fpn_shape[2], param->_count, param->_input_size, output_size);
    }
  }
  
  return 0;
}

template<typename T, int DLACHNUM>
__global__ void filterHWC_sigmoid(const T* in, float* out, int* devConuter, 
                                  float threashold, float scale, int maxNum, 
                                  int outChNum, int N) {
                                    
  int idx    = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = idx; i < N; i += stride) {
    int writePos = -1;
    T val = in[i * DLACHNUM];
    float valFP32 = getVal(val, scale);
    if(valFP32 > threashold) {
      writePos = atomicAdd(devConuter, 1);
      if(writePos < maxNum) {
        reinterpret_cast<int*>(out)[writePos * outChNum] = i;
        for(int j = 0; j < outChNum - 1; ++j) {
          int group = j / DLACHNUM;
          T value = in[group * N * DLACHNUM + i * DLACHNUM +  j % DLACHNUM];
          out[writePos * outChNum + j + 1] =  getVal(value, scale);
        }
      }      
    }
  }
}

// __global__ void fp162fp32(const __half* in, float* out, int H, int W, int N);

//For DLA fp16 chw16, similar to hwc if c<=16
extern "C" int ConfidenceFilterForLidarDLA(LidarFilterParam *param, cudaStream_t stream) {
  DPRINTF(2, "Lidar Filter Function Start\n");
  const int max_blocks = 8;
  const int max_threads = 256;                   
  float* output = (float*)param->outputs[0];
  int    outChNum = (param->fpn_shape[0] + 1);
  const int output_size = param->max_num * outChNum;
  float &_conf_thresh = param->thresholds[0];
  int _fpn_mode = param->mode;
  float threshold =  (_fpn_mode & 0x01) ? logf(_conf_thresh/(1-_conf_thresh)) : _conf_thresh;
  DPRINTF(2, "threshold is %.5lf\n", threshold);
  DPRINTF(2, "scale is %.5lf\n", param -> out_scale[0]);
  cudaMemsetAsync(param->outputs[0], 0x0CC, output_size * sizeof(float), stream);
  cudaMemsetAsync(param->_count[0],  0x0  , sizeof(int), stream);      

  { //FPNSum, for tfl2 header
    const int num_detections = param->fpn_shape[1] * param->fpn_shape[2];
    DPRINTF(2, "num_detections is : %d.\n", num_detections);
    dim3 grid;
    grid.x=max_blocks;
    grid.y=1;
    grid.z=1;
    if( 1 == param->data_type ){ // data_type = 1 fp16
      DPRINTF(2, "fp16 routine start \n");
      filterHWC_sigmoid<__half, 16><<<grid, max_threads, 0, stream>>>
      ((const __half*)param->_d_input_bufs[0], output, param->_count[0], threshold, param -> out_scale[0], param -> max_num,
       outChNum, num_detections);
    } else { // data_type = 2 int8
      DPRINTF(2, "int8 routine start \n");
      filterHWC_sigmoid<signed char, 32><<<grid, max_threads, 0, stream>>>
      ((const signed char*)param->_d_input_bufs[0], output, param->_count[0], threshold, param -> out_scale[0], param -> max_num,
       outChNum, num_detections);

    }
  }
  
  // int output_size = param->_input_size[0];
  // DPRINTF(1, "the output size in the DLA filter is : %d.\n", output_size);
  // cudaMemsetAsync(output, 0x0CC, output_size * sizeof(float), stream);
  // int H = param->fpn_shape[1];
  // int W = param->fpn_shape[2];
  // fp162fp32<<<max_blocks, max_threads, 0, stream>>>((const __half*)param->_d_input_bufs[0], output, H, W, output_size);

  
  return 0;
}


// __global__ void fp162fp32(const __half* in, float* out, int H, int W, int N) { 
//   int idx    = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x * gridDim.x;

//   for(int i = idx; i < N; i += stride) {
//     // int pos = i;
//     // int x = pos % 16;
//     // pos = pos - x;
//     // pos = pos / 16;

//     // int y = pos % 48;
//     // pos = pos - y;
//     // pos = pos / 48;

//     // int z = pos % 48;
//     // pos = pos - z;
//     // pos = pos / 48;

//     // int c = pos;

//     // int channel = c * 16 + x;
//     // int height  = z;
//     // int widht   = y;

//     int HW       = i % (H * W); //n
//     int channel  = i / (H * W); //C
//     int sChannel = channel % 16; 
//     int gChannel = channel / 16;

//     out[i] = __half2float(in[gChannel * (H * W) * 16 + HW * 16 + sChannel]);

//   }
// }