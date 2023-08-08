
#include "ConvLSTM.h"  

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
		  i += blockDim.x * gridDim.x)
		  
// CUDA: use 256 threads per block
const int CUDA_NUM_THREADS = 256;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

/* ConvLSTM replace a Uni-directional LSTM fully connected operation to a conv operation.
As describe in the paper: https://arxiv.org/abs/1506.04214 
  it = σ(Wxi * Xt + Whi * Ht�? + Wci �?Ct�? + bi)
  ft = σ(Wxf * Xt + Whf * Ht�? + Wcf �?Ct�? + bf)
  bt = tanh(Wxc * Xt + Whc * Ht�? + bc)
  Ct = ft �?Ct�? + it �?bt 
  ot = σ(Wxo * Xt + Who * Ht�? + Wco �?Ct + bo)
  Ht = ot �?tanh(Ct)

In TensorRT:
  _bottom = concat ( x, prev_h )
  _top = conv( _filter, _bottom )
  Ct = Sigmoid( ft ) * prev_c + Sigmoid( it ) * Tanh( bt )
  Ht = Sigmoid( ot ) * Tanh( Ct ) 
*/
__device__ float Sigmoid( const float x ) {
  return 1 / (1 + expf(-x));
}

__global__ void LSTM_kernel(int n, float *ft, float *it, float *bt, float *ot,
                            float* Ct, float *Ht, int ch, int w, int h, int dgbLevel = 0) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    float ct = Sigmoid( ft[i] ) * Ct[i] + Sigmoid( it[i] ) * tanhf( bt[i] );
    Ht[i] = Sigmoid( ot[i] ) * tanhf( ct );
    Ct[i] = ct;
    //if( dgbLevel>3  && i<16 ) printf("%d(%f,%f,%f) ", i, ft[i], Ht[i], Ct[i]);
    // if(i % (w * h) == 0 && i / (w * h) <= 16)
  }
}

__host__ __device__ float int8ToFloat(signed char val, float scale) { return (val * scale); }

__global__ void LSTM_kernel_int8(int n, signed char *ft, signed char *it, signed char *bt, signed char *ot,
                                 float* Ct, signed char *HtOut, signed char* Ht, float inScale, float outScaleLstm, float outScaleConcat,
                                 int ch, int w, int h,
                                 int dgbLevel = 0)
{
  CUDA_1D_KERNEL_LOOP(i, n) {

    float ftf = int8ToFloat(ft[i], inScale);
    float itf = int8ToFloat(it[i], inScale);
    float btf = int8ToFloat(bt[i], inScale);
    float otf = int8ToFloat(ot[i], inScale);
      
    float ct = Sigmoid(ftf) * Ct[i] + Sigmoid(itf) * tanhf(btf);
    float ht = Sigmoid(otf) * tanhf(ct);
    // HtOut[i] = static_cast<signed char>(__float2int_rn( ht * outScaleLstm));
    // Ht[i]    = static_cast<signed char>(__float2int_rn( ht * outScaleConcat));
    // Ht[i]    = ht;
    int hto = __float2int_rn( ht * outScaleLstm);
    int hti = __float2int_rn( ht * outScaleConcat);
    asm volatile( "cvt.pack.sat.s8.s32.b32   %0, 0, %1, 0;\n"
                  : "=r"(hto) : "r"(hto));
    HtOut[i] = static_cast<signed char>(hto);

    asm volatile( "cvt.pack.sat.s8.s32.b32   %0, 0, %1, 0;\n"
                  : "=r"(hti) : "r"(hti));
    Ht[i]    = hti;

    Ct[i]    = ct;


  }
}


__global__ void LSTM_Concat_InT8(signed char* in, signed char* dHt, signed char* out, float scale, float scaleHt, int n0, int n1, int n)
{

  signed char* input0 = in  + blockIdx.y * n0;
  signed char* input1 = dHt + blockIdx.y * n1;

  signed char* dConcat0 = out + blockIdx.y * (n0 + n1);
  signed char* dConcat1 = dConcat0 + n0;

  CUDA_1D_KERNEL_LOOP(i, n)
  {
    // dConcat0[i] = static_cast<signed char>(__float2int_rn(input0[i] * scale));
    int dC0 = __float2int_rn(input0[i] * scale);
    asm volatile( "cvt.pack.sat.s8.s32.b32   %0, 0, %1, 0;\n"
                  : "=r"(dC0) : "r"(dC0));
    dConcat0[i] = static_cast<signed char>(dC0);
    
    dConcat1[i] = input1[i];
    // dConcat1[i] = static_cast<signed char>(__float2int_rn(input1[i] / scaleHt));
  }
}


static int g_ConvLSTM_num = 0;
static std::unordered_map<cudaStream_t, std::unordered_map<int,void*>> g_ConvLSTM_buf;
static std::unordered_map<cudaStream_t, std::unordered_map<int,int>> g_ConvLSTM_size;
static std::unordered_map<cudaStream_t, std::unordered_map<int,float>> g_ConvLSTM_Ht_scale;

//state: 0:Restart frame, 1:other/normal frame
//ret: 0: normal, -1: not found
int SetConvLSTMState( cudaStream_t stream, int state ) {
  DPRINTF(2, "---------------------->state = %d\n", state);
  if( 0 == state) {
    if( g_ConvLSTM_buf.count(stream) == 0 ) { 
      return -1;
    }
    
    // Reset d_Ht/d_Ct buffer to 0.
    auto buf_map = g_ConvLSTM_buf[stream];
    auto size_map = g_ConvLSTM_size[stream];
    for ( auto it = buf_map.begin(); it != buf_map.end(); ++it ) {
      DPRINTF(2, "RESET g_ConvLSTM_buf[%d] = %p, with size: %d\n", it->first, it->second, size_map[it->first]);
      cudaMemsetAsync(it->second, 0, size_map[it->first], stream);
    }
  }
  return 0; 
}

int setConvLSTMBuff(void* ptr, cudaStream_t stream, int magic) {
  DPRINTF(2, "[%d]Free g_ConvLSTM_buf[%p][%d]=%p\n", g_ConvLSTM_num, stream, magic, ptr);
  cudaFree(ptr);
  ptr = nullptr;
  g_ConvLSTM_buf[stream][magic] = nullptr;

  return 0;
}

int ConvLSTMPluginForward(int batchSize, const void *const *inputs,void *const*outputs, void *workspace,
                            cudaStream_t stream,int params_int[2],void **params_pointer[2],Dims& input_dims,std::vector<int>&_attrs,
                            DataType mDataType, TensorFormat mTensorFormat, float mInputScale, float mOutputScale) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip ConvLSTMPlugin::enqueue!!\n");
    return 0;
  }

  int copyBytes = mDataType == DataType::kINT8 ? sizeof(char) : sizeof(float);
  
  int in_ch = _attrs[0];
  int hidden_ch = _attrs[1];
  int w = input_dims.d[2];
  int h = input_dims.d[1];
  int n = hidden_ch * w * h;  // elements size of d_Ht/d_Ct, per batch.
              
  int _mode = params_int[0];
  int _magic = params_int[1];
  void* &d_Ct = *params_pointer[0];
  void* &d_Ht = *params_pointer[1];
      
  DPRINTF(2, "ConvLSTMPlugin::enqueue _mode=%d, data type=%d, tensor format=%d\n", _mode, static_cast<int>(mDataType), static_cast<int>(mTensorFormat));
  DPRINTF(2, "ConvLSTMPlugin::enqueue in_ch=%d, hidden_ch=%d, w=%d, h=%d\n",in_ch, hidden_ch, w, h);
  if( 1 == _mode && nullptr == d_Ct ) { 
    d_Ct = d_Ht;
    g_ConvLSTM_buf[stream][_magic] = d_Ht;
    g_ConvLSTM_size[stream][_magic] =  batchSize * n * sizeof(float);
    g_ConvLSTM_Ht_scale[stream][_magic] = mOutputScale;
    DPRINTF(2, "[%d]Put g_ConvLSTM_buf[%p][%d]=%p\n", g_ConvLSTM_num, stream, _magic, d_Ht);
    DPRINTF(2, "[%d]Put g_ConvLSTM_size[%p][%d]=%d\n", g_ConvLSTM_num, stream, _magic, batchSize * n * sizeof(float));
    DPRINTF(2, "[%d]Put g_ConvLSTM_Ht_scale[%p][%d]=%.5f\n", g_ConvLSTM_num, stream, _magic, mOutputScale);
  } else if( 3 == _mode && nullptr == d_Ht ) { 
    d_Ht = g_ConvLSTM_buf[stream][_magic];
    g_ConvLSTM_buf[stream][_magic + 1] = d_Ct;
    g_ConvLSTM_size[stream][_magic + 1] = batchSize * n * sizeof(float);        
    DPRINTF(2, "[%d]Get g_ConvLSTM_buf[%p][%d]=%p Ct=%p, CT Size=%d\n", g_ConvLSTM_num, stream, _magic, d_Ht, d_Ct, batchSize * n * sizeof(float));
    g_ConvLSTM_num++;    
  }

  if(d_Ht == nullptr)
    CHECK_CUDA(cudaMalloc(&d_Ht, n * sizeof(float) * batchSize));

  if(d_Ct == nullptr)
    CHECK_CUDA(cudaMalloc(&d_Ct, n * sizeof(float) * batchSize));

  if (batchSize > 4) {
    batchSize = 4;
    DPRINTF(2,"Warning: ConvLSTMPlugin batchSize=%d. Must <=4\n", batchSize);
  }
  
  if( 1 == _mode ){
      // concat input & prev_h, just copy inbuf to d_input
      DPRINTF(2, "Mode 1: The input sacle is %.5f; output scale is %.5f.\n", mInputScale, mOutputScale);   

      if (mDataType == DataType::kFLOAT || (mDataType == DataType::kINT8 && fabs(mInputScale - mOutputScale) < 1e-3)) {
        DPRINTF(2, "ConvLSTM Copy Concat.\n");
        for( int nb=0; nb < batchSize; nb++) { 
          int n0 = in_ch * w * h;   // num of concat0
          int n1 = n;               // num of concat1, copy prev_h from d_Ht
          const char *d_in = (const char*)inputs[0] + nb * n0 * copyBytes;
          char *d_concat0 = (char*)outputs[0] + nb * (n0 + n1) * copyBytes;
          char *d_concat1 = d_concat0 + n0 * copyBytes;
          CHECK_CUDA(cudaMemcpyAsync(d_concat0, d_in, n0 * copyBytes, cudaMemcpyDeviceToDevice, stream));
          // Copy prev_h from d_Ht
          CHECK_CUDA(cudaMemcpyAsync(d_concat1, (char *)d_Ht  + nb * n1 * copyBytes,  n1 * copyBytes, cudaMemcpyDeviceToDevice, stream));
        }
      }else {
        float mOutputScaleHt= 1.f;
        // float mOutputScaleHt = g_ConvLSTM_Ht_scale[stream][_magic];
        DPRINTF(2, "ConvLSTM Kernel Concat.\n");
        int n0 = in_ch * w * h;   // num of concat0
        int n1 = n;               // num of concat1, copy prev_h from d_Ht
        float mNewScale = mInputScale / mOutputScale;
        dim3 concatGridDim(GET_BLOCKS(n), batchSize, 1);
        LSTM_Concat_InT8<<<concatGridDim, CUDA_NUM_THREADS, 0, stream>>>((signed char*)inputs[0], (signed char*)d_Ht, 
                                                                          (signed char*)outputs[0], mNewScale, mOutputScaleHt,
                                                                          n0, n1, n);

      }
      
  } else { //if( 3 == _mode )
    float mOutputScaleHt = 0.0f;
    mOutputScaleHt = mDataType == DataType::kINT8 ? g_ConvLSTM_Ht_scale[stream][_magic] : mOutputScaleHt;
    DPRINTF(2, "Mode 3 , Type: %d; The input sacle is %.5f;ht output scale is %.5f; Ht output scale is %.5f.\n", static_cast<int>(mDataType), mInputScale, mOutputScale, mOutputScaleHt);

    for( int nb = 0; nb < batchSize; nb ++) {

      signed char* d_top[4];
      for( int i = 0; i < 4; i++) {
        d_top[i] = ((signed char*)inputs[0]) + (nb * 4 + i) * n * copyBytes;
      }

      if(mDataType == DataType::kFLOAT) {
        LSTM_kernel<<<
          GET_BLOCKS(n),
          CUDA_NUM_THREADS,
          0, stream>>>(n, (float *)d_top[0], (float *)d_top[1], (float *)d_top[2], (float *)d_top[3],
                      ((float *)d_Ct) + nb * n, ((float *)d_Ht)  + nb * n, in_ch, w, h, TRT_DEBUGLEVEL);
      }else {
        LSTM_kernel_int8<<<
        GET_BLOCKS(n),
        CUDA_NUM_THREADS,
        0, stream>>>(n, d_top[0], d_top[1], d_top[2], d_top[3],
        ((float *)d_Ct) + nb * n, ((signed char *)outputs[0])  + nb * n, ((signed char *)d_Ht)  + nb * n,
        mInputScale, 1.0f / mOutputScale, 1.0f / mOutputScaleHt, in_ch, w, h, TRT_DEBUGLEVEL);        
      }
    }

    if(mDataType == DataType::kFLOAT) {
      //copy Ht to Output as float. Int8 needs write d_Ht to the output directly
      CHECK_CUDA(cudaMemcpyAsync(outputs[0], d_Ht, batchSize * n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
  }
      
  
  return 0;
}
