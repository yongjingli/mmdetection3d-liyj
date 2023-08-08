/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12 for retinaNet
 *
 * Input: class(5),box(5)
 * input -> Decode -> Concat -> NMS -> output
 */
#include <string.h>
#include "ConfidenceFilter.h"

// Static class fields initialization
PluginFieldCollection ConfidenceFilterV2Creator::mFC{};
std::vector<PluginField> ConfidenceFilterV2Creator::mPluginAttributes;

#define DEBUG_GPU_PRINT() //{if(m<4096 && 2>(m%16)||(m>=69312&&m<69312+16))printf("m:%d i:%d conf:%f\n", m, index[0], conf);}

__device__ float getVal(__half val) { return __half2float(val); }
__device__ float getVal(float val) { return val; }

//--- Kerenls for ConfidenceFilter, support mod/sod/rsm/tl header
template <typename T>
__global__ void filterNC_kernel(const int num_detections, const int max_num, 
                           const int conf_offset, const float threshold,
                           const T *input, const int col_in, float *output, int *count) {
  // Go through detections by descending score
  const int step = blockDim.x * gridDim.x; // 256 * 8  or 256 * 1
  const int col_out = (col_in+1);
  for (int m = blockDim.x * blockIdx.x + threadIdx.x; m < num_detections ; m+=step) {
    const T *box_ptr = input + m*col_in;
    //if(m<16 || (m>=6880&&m<6880+16) || (m>=31588&&m<31588+16 ) )
    //  printf("m:%d conf:%f\n", m, box_ptr[conf_offset]);
    if( getVal(box_ptr[conf_offset]) > threshold ) { // && count < max_in_block 
      int idx = atomicAdd( count, 1 );
      if( idx >= max_num ) break;

      ((int*)output)[idx*col_out] = m;
      float *out_ptr = output + idx*col_out + 1; 
      for( int i = 0 ; i< col_in; i++){
        out_ptr[i] = getVal(box_ptr[i]);
      }
      //if(idx<32)printf("f%d m:%d c:%f v:%f %f...%f\n", idx, m, box_ptr[conf_offset], out_ptr[0], out_ptr[1], out_ptr[col_in-1]);
    } 
  }
}

template <typename T>
__global__ void filterCHW_kernel(const int num_detections, const int max_num, 
                           const int conf_offset, const float threshold,
                           const T *input, const int col_in, float *output, int *count) {
  const T *conf_ptr = input + conf_offset * num_detections;
  const int step = blockDim.x * gridDim.x; // 256 * 8  or 256 * 1
  const int col_out = (col_in+1);
  for (int m = blockDim.x * blockIdx.x + threadIdx.x; m < num_detections ; m+=step) {
    //if(m<16||(m>=69312&&m<69312+16))printf("m:%d conf:%f\n", m, conf_ptr[m]);
    if( getVal(conf_ptr[m]) > threshold ) { // && count < max_in_block 
      int idx = atomicAdd( count, 1 );

      // if(idx == max_num)  printf("Warning: Beyond the limit!");
      if( idx >= max_num ) { 
        break;
      } 

      const T *box_ptr = input + m;
      ((int*)output)[idx*col_out] = m;
      float *out_ptr = output + idx*col_out + 1; 
      for( int i = 0 ; i< col_in; i++){
        out_ptr[i] = getVal(*box_ptr);
        box_ptr += num_detections;
      }
      //printf("f%d m:%d conf:%f value:%f \n", idx, m, out_ptr[0], out_ptr[1]);
    } 
  }
}

template <typename T>
__global__ void filterMultiCHW_kernel(const int num_detections, const int max_num, 
                           const int* conf_offsets, const float* thresholds,
                           const T *input, const int col_in, float *output, int *count,int offset_num) {
  
  const int step = blockDim.x * gridDim.x; // 256 * 8  or 256 * 1
  const int col_out = (col_in+1);
  for (int m = blockDim.x * blockIdx.x + threadIdx.x; m < num_detections ; m+=step) {
    bool save=false;
    for (int ii=0;ii<offset_num;ii++){
      const T *conf_ptr = input + conf_offsets[ii] * num_detections;
      if( getVal(conf_ptr[m]) > thresholds[ii] && save==false) { 
        int idx = atomicAdd( count, 1 );
        if( idx >= max_num ) break;
        save=true;
        const T *box_ptr = input + m;
        ((int*)output)[idx*col_out] = m;
        float *out_ptr = output + idx*col_out + 1; 
        for( int i = 0 ; i< col_in; i++){
          out_ptr[i] = getVal(*box_ptr);
          box_ptr += num_detections;
        }
      } 
    }
 
  }
}

template <int PAD_LEFT, int PAD_TOP, int FILTER_SUM_NUM, typename T>
__global__ void filter_FpnSum_kernel(const int num_detections, const int4 *input_dims, const int max_num, 
                           const int conf_offset, const float threshold,
                           const T **input, const int col_in, float *output, 
                           const int fpn_w, int *count, int *input_size, int output_size) {
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
    const int x_o = m % fpn_w + PAD_LEFT;
    const int y_o = m / fpn_w + PAD_TOP;
    int index[3];
    float conf=0.f;
    for( int i=0; i<FILTER_SUM_NUM; i++) {
      const int x_i = x_o / input_dim_s[i].z;
      const int y_i = y_o / input_dim_s[i].z;
      const int ch_i = x_o % input_dim_s[i].z + (y_o % input_dim_s[i].z)*input_dim_s[i].z;
      index[i] = ( ch_i * input_dim_s[i].y + y_i) * input_dim_s[i].x + x_i;
      conf += getVal(conf_ptr_s[i][index[i]]);
    }
    DEBUG_GPU_PRINT();
    if( conf > threshold ) { // && count < max_in_block 
      int idx = atomicAdd( count+blockIdx.y, 1 );
      if( idx >= max_num ) continue;

      float *out_ptr = (output+blockIdx.y*output_size) + idx*col_out; 
      ((int*)out_ptr)[0] = m;
      out_ptr[1] = conf;        
      for( int k = 1 ; k < col_in; k++){
        if( 1 == FILTER_SUM_NUM ){      
          out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index[0]]);
        } else {
          out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index[0]])
                       + getVal(conf_ptr_s[1][input_dim_s[1].w * k + index[1]])
                       + getVal(conf_ptr_s[2][input_dim_s[2].w * k + index[2]]);
        }                                                    
      } 
      //printf("f%d m:%d conf:%f value:%f\n",blockIdx.y, idx, m, conf, out_ptr[2]);
    } 
  }
  //if(blockIdx.x==0 && threadIdx.x==0 )
  //printf("filter_FpnSum_kernel done\n");
}

//for bev model and mod3d(cngpV7.x)
template <int FILTER_SUM_NUM, typename T>
__global__ void filter_FpnSum_BEVPooling_kernel(const int num_detections, const int4 *input_dims, const int max_num, 
                           const int* conf_offsets, const float* thresholds, const int conf_num,
                           const T **input, const int col_in, float *output, 
                           const int fpn_w,const int fpn_h,  int *count, int *input_size, int output_size) {
  
  const int x_o=threadIdx.x+blockDim.x*blockIdx.x;//0-(180/BEV_POOLING_SIZE)
  const int y_o=threadIdx.y+blockDim.y*blockIdx.y;//0-(540/BEV_POOLING_SIZE)                         
  __shared__ int4 input_dim_s[FILTER_SUM_NUM];
  __shared__ const T* conf_ptr_s[FILTER_SUM_NUM];
  if(threadIdx.y==0 && threadIdx.x < FILTER_SUM_NUM){
    input_dim_s[threadIdx.x] = input_dims[threadIdx.x];
    conf_ptr_s[threadIdx.x] = input[threadIdx.x]+blockIdx.z*input_size[threadIdx.x];;
  }
  __syncthreads();
  const int col_out = (col_in+1); 
  if(x_o<fpn_w/BEV_POOLING_SIZE && y_o<fpn_h/BEV_POOLING_SIZE){
    int index[FILTER_SUM_NUM];
    const int x_oo=BEV_POOLING_SIZE*x_o;
    const int y_oo=BEV_POOLING_SIZE*y_o;
    for(int a=0;a<conf_num;a++){
      int m=0;
      float conf_max=thresholds[a];
      bool save=false;      
      for(int hh=0;hh<BEV_POOLING_SIZE;hh++ ){
        for(int ww=0;ww<BEV_POOLING_SIZE;ww++){
          float conf_attrib_tmp=0;
          int  index_i[FILTER_SUM_NUM];
          for( int i=0; i<FILTER_SUM_NUM; i++) {
            const int x_i = (x_oo+ww) / input_dim_s[i].z;
            const int y_i = (y_oo+hh) / input_dim_s[i].z;
            const int ch_i = (x_oo+ww) % input_dim_s[i].z + ((y_oo+hh) % input_dim_s[i].z)*input_dim_s[i].z;
            index_i[i] = ( ch_i * input_dim_s[i].y + y_i) * input_dim_s[i].x + x_i;
            conf_attrib_tmp += getVal(conf_ptr_s[i][input_dim_s[i].w * conf_offsets[a]+index_i[i]]);
          }
          //if(a!=1 && conf_attrib_tmp>0.5) printf("conf:%f class:%d\n",conf_attrib_tmp, a);
          if( conf_attrib_tmp>conf_max) {
            save = true;
            conf_max=conf_attrib_tmp;
            index[0]=index_i[0],index[1]=index_i[1],index[2]=index_i[2];
            m=(y_oo+hh)*fpn_w+(x_oo+ww);
          }
        }
      }
      //save the output
      if( save ) { 
        int idx = atomicAdd( count+blockIdx.z*(conf_num+1), 1 );
        if(idx < max_num) {
          //if(idx>500) printf("%d attrib_idx:%d class:%d\n",idx, idx_attrib,attrib);
          float *out_ptr = (output+blockIdx.z*output_size) + idx*col_out; 
          ((int*)out_ptr)[0] = m;       
          for( int k = 0 ; k < col_in; k++){
            if( 1 == FILTER_SUM_NUM ){      
              out_ptr[k+1] = conf_ptr_s[0][input_dim_s[0].w * k + index[0]];
            } else {
              out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index[0]])
                           + getVal(conf_ptr_s[1][input_dim_s[1].w * k + index[1]])
                           + getVal(conf_ptr_s[2][input_dim_s[2].w * k + index[2]]);
            }
          } 
        }
      }
    }
  }
}

//For Mono3d task
template <int PAD_LEFT, int PAD_TOP, int FILTER_SUM_NUM, typename T>
__global__ void filter_mono3d_kernel(const int num_detections, const int4 *input_dims, const int max_num, 
                           const float threshold, const int conf_num, const T **input, const int col_in, float *output, 
                           const int fpn_w,const int fpn_h, int *count, int *input_size, int output_size) {
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
    const int x_o = m % fpn_w + PAD_LEFT;
    const int y_o = m / fpn_w + PAD_TOP;
    const int x_i = x_o / input_dim_s[0].z;
    const int y_i = y_o / input_dim_s[0].z;
    const int ch_i = x_o % input_dim_s[0].z + (y_o % input_dim_s[0].z)*input_dim_s[0].z;
    int index = ( ch_i * input_dim_s[0].y + y_i) * input_dim_s[0].x + x_i;
    for( int n=0; n<conf_num;n++){
      float conf = getVal(conf_ptr_s[0][input_dim_s[0].w * n + index]);

      //if(m < 16)printf("m:%d n:%d conf:%f\n", m, n, conf);
      if( conf > threshold ) { // && count < max_in_block 
        int idx = atomicAdd( count+blockIdx.y, 1 );
        if( idx >= max_num ) return;
  
        float *out_ptr = (output+blockIdx.y*output_size) + idx*col_out; 
        ((int*)out_ptr)[0] = m;        
        for( int k = 0 ; k < col_in; k++){    
          out_ptr[k+1] = getVal(conf_ptr_s[0][input_dim_s[0].w * k + index]);
        } 
        //printf("No.%d m:%d n:%d conf:%f value:%f\n",idx, m, n, conf, out_ptr[7]);
        break;        
      }
    }
  }
}

template <typename T>
__global__ void filter_FpnConcat_kernel(const int4 *input_dims, const int max_num, 
                           const int conf_offset, const float threshold,
                           const T **input, const int col_in, float *output, int *count,
                           int *input_size, int output_size) {
  //__shared__ int count; 
  __shared__ int4 input_dim;
  __shared__ const T* conf_ptr;
  __shared__ int num_detections;
  __shared__ int val_blocksize;
  const int ratio = 2;
  __shared__ int ra_c;
  if(0==threadIdx.x){
    input_dim = input_dims[blockIdx.x];
    num_detections = abs(input_dim.z) * input_dim.y * input_dim.x; // 240,870,1450,3480,5800,5800
    val_blocksize = ((input_dim.z>0)?1:4) * input_dim.y * input_dim.x;
    conf_ptr = (input[blockIdx.x]+blockIdx.y*input_size[blockIdx.x]) + val_blocksize * conf_offset;
    ra_c = input_dim.z / (-ratio*ratio);
    //printf("blockIdx:%d num_detections:%d conf:%p\n", blockIdx.x, num_detections, conf_ptr);
  }
  __syncthreads();

  // Go through detections by descending score
  const int step = blockDim.x; // 256 
  const int col_out = (col_in+1);
  for (int m = threadIdx.x; m < num_detections ; m+=step) {
    int index;
    if( input_dim.z > 0 ) {
      const int x_i = m / input_dim.z;
      const int ch_i = m % input_dim.z;
      index = (ch_i * col_in) * input_dim.y * input_dim.x + x_i;
    } else { // == -4, -8
      const int m_i = m % ra_c;
      const int m_o = m / ra_c;
      const int out_w = input_dim.x * ratio;
      const int x_o = m_o % out_w;
      const int y_o = m_o / out_w;
      const int x_i = x_o / ratio;
      const int y_i = y_o / ratio;
      const int ch_i = (m_i*col_in*ratio+(y_o % ratio))*ratio + (x_o % ratio);
      index = ( ch_i * input_dim.y + y_i) * input_dim.x + x_i;
    }

    //const int mo = m + input_dim.w;
    //if(mo<16 || (mo>=6880&&mo<6880+16) || (mo>=31588&&mo<31588+16 ) )
    //  printf("%d m:%d conf:%f index:%d \n", m, mo, conf_ptr[index], index);
    if( getVal(conf_ptr[index]) > threshold ) { // && count < max_in_block
      int idx = atomicAdd( count+blockIdx.y, 1 );
      if( idx >= max_num ) continue;

      const T *box_ptr = (input[blockIdx.x]+blockIdx.y*input_size[blockIdx.x]) + index;
      float *out_ptr = (output+blockIdx.y*output_size) + idx*col_out; 
      ((int*)out_ptr)[0] = m + input_dim.w;
      for( int i = 1 ; i< col_out; i++){
        out_ptr[i] = getVal(*box_ptr);
        box_ptr += val_blocksize;
      }
      //printf("bathc:%d f%d m:%d c:%f v:%f %f...%f\n",blockIdx.y, idx, m + input_dim.w, conf_ptr[index], out_ptr[0], out_ptr[1], out_ptr[col_in-1]);
    } 
  }
  //if(blockIdx.x==0 && threadIdx.x==0 )
  //printf("filter_FpnConcat_kernel done\n");
}

int ConfidenceFilterForward(int batchSize, const void *const *inputs,void *const*outputs, void *worksapce,
                            cudaStream_t stream,int *params,int* params_int_pointer[5],std::vector<int>& _conf_offsets,
                            std::vector<Dims>&_input_dims,const void **_d_input_bufs, const DataType mInDataType, const TensorFormat mInTensorFormat) {

  
  int _max_num=params[0];
  int _mode=params[1];
  int _input_num=params[2];
  int _ch=params[3];
  int _max_batch_size=params[4];
  int _fpn_mode=params[5];
  int _conf_offset=params[6];
  float _conf_thresh=(float)params[7]/100;
  int _fpn_shape[3] = {params[8],params[9],params[10]}; // fpn shape: ch, fpn_h, fpn_w

  int *_d_input_dims = params_int_pointer[0];
  int *_d_input_size = params_int_pointer[1];
  int *_d_count = params_int_pointer[2];
  int *_d_conf_offsets = params_int_pointer[3];
  float *_d_conf_thresholds = (float *)params_int_pointer[4]; 

  // Launch actual Filter kernel - 8 block with each thread handling n item
  const int max_blocks = (_max_num>=64)?8:1;
  const int max_threads = 256;
  float threshold =  (_mode & 0x01) ? logf(_conf_thresh/(1-_conf_thresh)) : _conf_thresh;
  DPRINTF(3, "ConfidenceFilter in=%d _mode=0x%x threshold =  %f\n", _input_num, _mode, threshold);
  const int col_num = _ch;
  const int conf_num = _conf_offsets.size();  
  const int output_size = _max_num*(col_num+1);
  // 0xCC -> float:-1.07374176E8 int:-858993460
  cudaMemsetAsync(outputs[0], 0x0CC, batchSize*output_size*sizeof(float), stream);
  cudaMemsetAsync(_d_count, 0x0, _max_batch_size*_input_num*sizeof(int)*(MAX_OFFSET_NUM+1), stream);
  if ( 0 == _fpn_mode ) {  // not FPN, single one input 
    auto const &input0_dims = _input_dims[0];
    int num_detections = input0_dims.d[0]; // NC
    if (3 == input0_dims.nbDims ) {  // CHW
      num_detections = input0_dims.d[1] * input0_dims.d[2];
    } 
    const int input_size = num_detections * col_num;
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
      const void* input;
      if (mInDataType == DataType::kFLOAT) {
        input = (const float*)inputs[0] + batchIdx * input_size;
      } else {
        input = (const __half*)inputs[0] + batchIdx * input_size;
      }
      float* output = (float*)outputs[0] + batchIdx * output_size;

      if (3 == input0_dims.nbDims ) {  // CHW
        if (_conf_offsets.size()>1){
          if (mInTensorFormat == TensorFormat::kLINEAR) {
            if (mInDataType == DataType::kFLOAT) {
              filterMultiCHW_kernel<<<max_blocks, max_threads, 0, stream>>>( num_detections,_max_num, 
                  _d_conf_offsets, _d_conf_thresholds, (float *)input, col_num, output, _d_count+batchIdx,_conf_offsets.size());
            } else {
              filterMultiCHW_kernel<<<max_blocks, max_threads, 0, stream>>>( num_detections,_max_num, 
                  _d_conf_offsets, _d_conf_thresholds, (__half *)input, col_num, output, _d_count+batchIdx,_conf_offsets.size());
            }
          }
        }else{
          if (mInTensorFormat == TensorFormat::kLINEAR) {
            if (mInDataType == DataType::kFLOAT) {
              filterCHW_kernel<<<max_blocks, max_threads, 0, stream>>>(num_detections, _max_num, 
                  _conf_offset, threshold, (float *)input, col_num, output, _d_count+batchIdx);
            } else {
              filterCHW_kernel<<<max_blocks, max_threads, 0, stream>>>(num_detections, _max_num, 
                  _conf_offset, threshold, (__half *)input, col_num, output, _d_count+batchIdx);
            }
          }
        }
      } else { // N * C
        if (mInTensorFormat == TensorFormat::kLINEAR) {
          if (mInDataType == DataType::kFLOAT) {
            filterNC_kernel<<<max_blocks, max_threads, 0, stream>>>(num_detections, _max_num, 
                _conf_offset, threshold, (float *)input, col_num, output, _d_count+batchIdx);
          } else {
            filterNC_kernel<<<max_blocks, max_threads, 0, stream>>>(num_detections, _max_num, 
                _conf_offset, threshold, (__half *)input, col_num, output, _d_count+batchIdx);
          }
        }
      }
    }
  } else { // FPN, mulity input
    
      float* output = (float*)outputs[0];
      for( int i=0; i< _input_num; i++){
          _d_input_size[i] = _input_dims[i].d[0] * _input_dims[i].d[1] * _input_dims[i].d[2];
          _d_input_bufs[i] = (const void*)inputs[i] ;
      } 
      if (0x20 == _fpn_mode)  { //FPNConcat, for mod/sod header
        dim3 grid;
        grid.x=_input_num;
        grid.y=batchSize;
        grid.z=1;
        if (mInTensorFormat == TensorFormat::kLINEAR) {
          if (mInDataType == DataType::kFLOAT) {
            filter_FpnConcat_kernel<<<grid, max_threads, 0, stream>>>((int4*)_d_input_dims, _max_num, 
              _conf_offset, threshold, (const float **)_d_input_bufs, col_num, output, _d_count,_d_input_size,output_size);
          } else if (mInDataType == DataType::kHALF) {
            filter_FpnConcat_kernel<<<grid, max_threads, 0, stream>>>((int4*)_d_input_dims, _max_num, 
              _conf_offset, threshold, (const __half **)_d_input_bufs, col_num, output, _d_count,_d_input_size,output_size);
          }
        } else if (mInTensorFormat == TensorFormat::kCHW2) {
        }
      } else { //FPNSum, for tl header
        const int num_detections = _fpn_shape[1] * _fpn_shape[2];
        dim3 grid;
        grid.x=max_blocks;
        grid.y=batchSize;
        grid.z=1;
        if( 0x10 == _fpn_mode && 457 == _fpn_shape[2] ) { 
			    if (mInTensorFormat == TensorFormat::kLINEAR) {
            if (mInDataType == DataType::kFLOAT) {
              filter_FpnSum_kernel<3,1,1><<<grid, max_threads, 0, stream>>>(
                num_detections, (const int4*)_d_input_dims, _max_num, 
                _conf_offset, threshold, (const float **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
            } else if (mInDataType == DataType::kHALF) {
              filter_FpnSum_kernel<3,1,1><<<grid, max_threads, 0, stream>>>(
                num_detections, (const int4*)_d_input_dims, _max_num, 
                _conf_offset, threshold, (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
            }
          } else if (mInTensorFormat == TensorFormat::kCHW2) {
          }
        } else {
          if(conf_num>1){
            if(1 == _input_num && 3 == conf_num && _conf_offsets[1] == _conf_offsets[2]) { // continuously confidence offset: [begin, end), length
              DPRINTF(3, "ConfidenceFilter filter_mono3d_kernel\n");                                     // Support: begin == 0, end == length
              if (mInTensorFormat == TensorFormat::kLINEAR) {
                if (mInDataType == DataType::kFLOAT) {
                  filter_mono3d_kernel<0,0,1><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, threshold, _conf_offsets[2],
                    (const float **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                } else if (mInDataType == DataType::kHALF) {
                  filter_mono3d_kernel<0,0,1><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, threshold, _conf_offsets[2],
                    (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                }
              } else if (mInTensorFormat == TensorFormat::kCHW2) {
              }              
            } else if(3 == _input_num || 1 == _input_num) { // for bev model and mod3d(cngpV7)
              dim3 block(16,16);
              grid.x= (_fpn_shape[2]/BEV_POOLING_SIZE-1)/block.x+1;
              grid.y= (_fpn_shape[1]/BEV_POOLING_SIZE-1)/block.y+1;
              grid.z= batchSize;
              if( 3 == _input_num ){
                if (mInTensorFormat == TensorFormat::kLINEAR) {
                  if (mInDataType == DataType::kFLOAT) {
                    filter_FpnSum_BEVPooling_kernel<3><<<grid, block, 0, stream>>>(
                      num_detections, (const int4*)_d_input_dims, _max_num, 
                      _d_conf_offsets, _d_conf_thresholds, conf_num, (const float **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                  } else if (mInDataType == DataType::kHALF) {
                    filter_FpnSum_BEVPooling_kernel<3><<<grid, block, 0, stream>>>(
                      num_detections, (const int4*)_d_input_dims, _max_num, 
                      _d_conf_offsets, _d_conf_thresholds, conf_num, (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                  }
                } else if (mInTensorFormat == TensorFormat::kCHW2) {
                }
              } else { // input_num = 1
                if (mInTensorFormat == TensorFormat::kLINEAR) {
                  if (mInDataType == DataType::kFLOAT) {
                    filter_FpnSum_BEVPooling_kernel<1><<<grid, block, 0, stream>>>(
                      num_detections, (const int4*)_d_input_dims, _max_num, 
                      _d_conf_offsets, _d_conf_thresholds, conf_num, (const float **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                  } else if (mInDataType == DataType::kHALF) {
                    filter_FpnSum_BEVPooling_kernel<1><<<grid, block, 0, stream>>>(
                      num_detections, (const int4*)_d_input_dims, _max_num, 
                      _d_conf_offsets, _d_conf_thresholds, conf_num, (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2],_fpn_shape[1], _d_count,_d_input_size,output_size);
                  }
                } else if (mInTensorFormat == TensorFormat::kCHW2) {
                }
              }
            } else {                 
               printf("Not support input=%d! Skip ConfidenceFilter::enqueue!!\n", _input_num);   
            }
          }else{
            if( 1 == _input_num ) {
              DPRINTF(3, "ConfidenceFilter filter_FpnSum_kernel\n");
  					  if (mInTensorFormat == TensorFormat::kLINEAR) {
                if (mInDataType == DataType::kFLOAT) {
                  filter_FpnSum_kernel<0,0,1><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, 
                    _conf_offset, threshold, (const float **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
                } else if (mInDataType == DataType::kHALF) {
                  filter_FpnSum_kernel<0,0,1><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, 
                    _conf_offset, threshold, (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
                }
              } else if (mInTensorFormat == TensorFormat::kCHW2) {
              }
            } else { // input_num == 3
              if (mInTensorFormat == TensorFormat::kLINEAR) {
                if (mInDataType == DataType::kFLOAT) {
                  filter_FpnSum_kernel<0,0,3><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, 
                    _conf_offset, threshold, (const float **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
                } else if (mInDataType == DataType::kHALF) {
                  filter_FpnSum_kernel<0,0,3><<<grid, max_threads, 0, stream>>>(
                    num_detections, (const int4*)_d_input_dims, _max_num, 
                    _conf_offset, threshold, (const __half **)_d_input_bufs, col_num, output, _fpn_shape[2], _d_count,_d_input_size,output_size);
                }
              } else if (mInTensorFormat == TensorFormat::kCHW2) {
              }
            }
          }
        }
    }  
  }
  return 0;
}
