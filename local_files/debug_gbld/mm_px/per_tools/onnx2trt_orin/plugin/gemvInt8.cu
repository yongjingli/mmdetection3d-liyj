
#include "gemvInt8.h"

#define THREAD 64

// Static class fields initialization
PluginFieldCollection GemvInt8PluginV2Creator::mFC{};
std::vector<PluginField> GemvInt8PluginV2Creator::mPluginAttributes;

__device__ float dotInt8(KTYPE4 a, float4 b) {
  return (float(a.x) * (b.x) + float(a.y) * (b.y) + float(a.z) * (b.z) +
          float(a.w) * (b.w));
}

// cuda kernel for gemv, Int8 weight
__global__ void gemvInt8(int m, int qn, KTYPE4 *kernel, float *bias,
                         float *scale, float4 *d_in, float *d_out) {
  int i;
  int div = qn / THREAD;
  __shared__ float tmp[THREAD];

  tmp[threadIdx.x] = 0.0;

  for (i = 0; i < div; i++) {
    tmp[threadIdx.x] +=
        dotInt8(kernel[blockIdx.x * qn + i * THREAD + threadIdx.x],
                d_in[i * THREAD + threadIdx.x]);
  }

  if (threadIdx.x < (qn % THREAD)) {
    tmp[threadIdx.x] +=
        dotInt8(kernel[blockIdx.x * qn + THREAD * div + threadIdx.x],
                d_in[THREAD * div + threadIdx.x]);
  }

  __syncthreads();

  for (i = THREAD / 2; i > 31; i = i / 2) {
    if (threadIdx.x < i) tmp[threadIdx.x] += tmp[threadIdx.x + i];
    __syncthreads();
  }

  if (threadIdx.x < 16) {
    tmp[threadIdx.x] += tmp[threadIdx.x + 16];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 8];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 4];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 2];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    d_out[blockIdx.x] = tmp[0] * scale[blockIdx.x] + bias[blockIdx.x];
}

// cuda kernel for gemv, Int8 weight, batch = 2
__global__ void gemvInt8b2(int m, int qn, KTYPE4 *kernel, float *bias,
                           float *scale, float4 *d_in, float4 *d_in2,
                           float *d_out) {
  int i;
  int div = qn / THREAD;
  __shared__ float tmp[THREAD];
  __shared__ float tmp2[THREAD];

  tmp[threadIdx.x] = 0.0;
  tmp2[threadIdx.x] = 0.0;

  for (i = 0; i < div; i++) {
    char4 &kval = kernel[blockIdx.x * qn + i * THREAD + threadIdx.x];
    tmp[threadIdx.x] += dotInt8(kval, d_in[i * THREAD + threadIdx.x]);
    tmp2[threadIdx.x] += dotInt8(kval, d_in2[i * THREAD + threadIdx.x]);
  }

  if (threadIdx.x < (qn % THREAD)) {
    char4 &kval = kernel[blockIdx.x * qn + div * THREAD + threadIdx.x];
    tmp[threadIdx.x] += dotInt8(kval, d_in[div * THREAD + threadIdx.x]);
    tmp2[threadIdx.x] += dotInt8(kval, d_in2[div * THREAD + threadIdx.x]);
  }

  __syncthreads();

  for (i = THREAD / 2; i > 31; i = i / 2) {
    if (threadIdx.x < i) {
      tmp[threadIdx.x] += tmp[threadIdx.x + i];
      tmp2[threadIdx.x] += tmp2[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (threadIdx.x < 16) {
    tmp[threadIdx.x] += tmp[threadIdx.x + 16];
    tmp2[threadIdx.x] += tmp2[threadIdx.x + 16];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 8];
    tmp2[threadIdx.x] += tmp2[threadIdx.x + 8];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 4];
    tmp2[threadIdx.x] += tmp2[threadIdx.x + 4];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 2];
    tmp2[threadIdx.x] += tmp2[threadIdx.x + 2];
    __syncthreads();
    tmp[threadIdx.x] += tmp[threadIdx.x + 1];
    tmp2[threadIdx.x] += tmp2[threadIdx.x + 1];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_out[blockIdx.x] = tmp[0] * scale[blockIdx.x] + bias[blockIdx.x];
    d_out[blockIdx.x + m] = tmp2[0] * scale[blockIdx.x] + bias[blockIdx.x];
  }
}

#define THREAD2 (THREAD)

// cuda kernel for gemv, Int8 weight, using fp16
__device__ void dotInt8fp16(__half2 &c2, char a, __half2 b2) {
  c2 = __hfma2(__half2half2(__short2half_rd(short(a))), b2, c2);
}

// cuda kernel for gemv, Int8 weight, using fp16: 74ms , fp32: 79ms. Overflow
__global__ void gemvInt8fp16(int m, int n, char *kernel, float *bias,
                             float *scale, float *d_infp32, __half2 *d_in,
                             float *d_out) {
  int i;
  int div = n / THREAD2;
  __shared__ __half2 tmp[THREAD2];

  {
    for (i = blockIdx.x; i < div; i += m) {
      unsigned int inoffset = threadIdx.x + i * THREAD2;
      d_in[inoffset] = __floats2half2_rn(d_infp32[inoffset] / 1e3,
                                         d_infp32[n + inoffset] / 1e3);
    }
  }

  __threadfence();

  tmp[threadIdx.x] = {0.0, 0.0};

  unsigned int offset = threadIdx.x;
  for (i = 0; i < div; i++) {
    dotInt8fp16(tmp[threadIdx.x], kernel[blockIdx.x * n + offset],
                d_in[offset]);
    offset += THREAD2;
  }

  __syncthreads();

  for (i = THREAD2 / 2; i > 31; i = i / 2) {
    if (threadIdx.x < i)
      tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + i]);
    __syncthreads();
  }

  if (threadIdx.x < 16) {
    tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + 16]);
    __syncthreads();
    tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + 8]);
    __syncthreads();
    tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + 4]);
    __syncthreads();
    tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + 2]);
    __syncthreads();
    tmp[threadIdx.x] = __hadd2(tmp[threadIdx.x], tmp[threadIdx.x + 1]);
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    d_out[blockIdx.x] =
        __low2float(tmp[0]) * scale[blockIdx.x] * 1e3 + bias[blockIdx.x];
    d_out[blockIdx.x + m] =
        __high2float(tmp[0]) * scale[blockIdx.x] * 1e3 + bias[blockIdx.x];
  }
}



// interface for Plugin


int GemvInt8PluginV2Forward(int batchSize, const void *const *inputs,
                            void *const*outputs, void *workspace,
                            cudaStream_t stream,int m,int qn,KTYPE *_d_kernel,
                            float *_d_bias,float *_d_scale) {
                              
  cudaDebugError(2);
  DPRINTF(3, "GemvInt8PluginV2::enqueue m=%d, n/4=%d, b=%d\n", m, qn, batchSize);
  
  if (2 == batchSize) {
#if 1
    gemvInt8b2<<<m, THREAD, 0, stream>>>(
        m, qn, (KTYPE4 *)_d_kernel, _d_bias, _d_scale, (float4 *)inputs[0],
        (float4 *)inputs[0] + qn, (float *)outputs[0]);
#else  // TODO: fp16 overflow
    gemvInt8fp16<<<m, THREAD2, 0, stream>>>(
        m, qn * 4, (KTYPE *)_d_kernel, _d_bias, _d_scale, (float *)inputs[0],
        (__half2 *)workspace, (float *)outputs[0]);
#endif

  } else {
    for (int nb = 0; nb < batchSize; nb++) {
      gemvInt8<<<m, THREAD, 0, stream>>>(
          m, qn, (KTYPE4 *)_d_kernel, _d_bias, _d_scale,
          (float4 *)inputs[0] + nb * qn, (float *)outputs[0] + nb * m);
    }
  }

  cudaDebugError(2);
  return cudaGetLastError() != cudaSuccess;;
}
