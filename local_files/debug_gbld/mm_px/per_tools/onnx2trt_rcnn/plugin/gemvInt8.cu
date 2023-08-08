#include <cuda_fp16.h>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include "MaskrcnnPlugin.hpp"

#define CHECK_CUDA(call)         \
  do {                           \
    cudaError_t status = call;   \
    if (status != cudaSuccess) { \
      return status;             \
    }                            \
  } while (0)

#define THREAD 64

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

inline std::vector<int> partsort_indexes(const float v[], int vsize,
                                         int partsort = 0) {
  // initialize original index locations
  std::vector<int> idx(vsize);
  for (int i = 0; i < vsize; ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  std::partial_sort(idx.begin(), idx.begin() + partsort, idx.end(),
                    [&v](int i1, int i2) { return fabs(v[i1]) > fabs(v[i2]); });
  idx.resize(partsort);

  return idx;
}

inline void calInt8Weight(float *value, int rows, std::vector<KTYPE> &kernels,
                          std::vector<float> &scales) {
  char *val = getenv("TRT_GEMV");
  int partNum = 0;
  if (NULL != val) {
    partNum = atoi(val);
    DPRINTF(2, "getenv TRT_GEMV partNum = %d\n", partNum);
    if (partNum <= 0) return;
  }

  int offset = 0;
  int n = kernels.size() / rows;
  DPRINTF(3, "Gemv m=%d, n=%d\n", rows, n);
  for (int m = 0; m < rows; m++) {
    auto idx = partsort_indexes((value + offset), n, partNum);
    float maxVal = fabs(value[offset + idx[partNum - 1]]);
    float scale = (maxVal + 1e-7) / 127.f;
    DPRINTF(3, "%d Gemv [%f,%f...] scale %f maxVal=%f\n", m, value[offset],
            value[offset + 1], scale, maxVal);
    scales[m] = scale;
    for (int i = 0; i < n; i++, offset++) {
      int tmpval = (int)round(value[offset] / scale);
      kernels[offset] =
          (tmpval > 127) ? 127 : ((tmpval < -128) ? -128 : tmpval);
    }
  }
}

// interface for Plugin
GemvInt8Plugin::GemvInt8Plugin(int rows, nvinfer1::Weights const &kernel,
                               nvinfer1::Weights const &bias)
    : _nrow(rows) {
  assert(rows == bias.count);
  if (bias.type == nvinfer1::DataType::kFLOAT) {
    _h_bias.assign((float *)bias.values, (float *)bias.values + bias.count);
    _h_scale.assign((float *)bias.values, (float *)bias.values + bias.count);
  } else {
    throw std::runtime_error("Unsupported bias dtype");
  }

  if (kernel.type == nvinfer1::DataType::kFLOAT) {
    _h_kernel.assign((KTYPE *)kernel.values,
                     (KTYPE *)kernel.values + kernel.count);
    calInt8Weight((float *)kernel.values, rows, _h_kernel, _h_scale);
  } else {
    throw std::runtime_error("Unsupported kernel dtype");
  }
}

int GemvInt8Plugin::initialize() {
  if (_initialized) {
    return 0;
  }
  DPRINTF(2, "GemvInt8Plugin _nrow %d _h_kernel=%ld\n", _nrow,
          _h_kernel.size());

  // calInt8Weight(_h_kernel.data(), _nrow, _h_kernel, _h_scale ); //for
  // optimaztion

  size_t nkernel_bytes = _h_kernel.size() * sizeof(KTYPE);
  size_t nbias_bytes = _nrow * sizeof(float);
  CHECK_CUDA(cudaMalloc((void **)&_d_kernel, nkernel_bytes));
  CHECK_CUDA(cudaMalloc((void **)&_d_bias, nbias_bytes));
  CHECK_CUDA(cudaMalloc((void **)&_d_scale, nbias_bytes));
  CHECK_CUDA(cudaMemcpy(_d_kernel, _h_kernel.data(), nkernel_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(_d_bias, _h_bias.data(), nbias_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(_d_scale, _h_scale.data(), nbias_bytes,
                        cudaMemcpyHostToDevice));

  _initialized = true;
  return 0;
}

void GemvInt8Plugin::terminate() {
  if (!_initialized) {
    return;
  }
  cudaFree(_d_scale);
  cudaFree(_d_bias);
  cudaFree(_d_kernel);
  _initialized = false;
}

int GemvInt8Plugin::enqueue(int batchSize, const void *const *inputs,
                            void **outputs, void *workspace,
                            cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip GemvInt8Plugin::enqueue!!\n");
    return 0;
  }

  assert(_initialized);
  nvinfer1::Dims input_dims = this->getInputDims(0);
  int qn = input_dims.d[0] / 4;  // quarter of n ( as float4 )
  int m = _nrow;
  DPRINTF(3, "GemvInt8Plugin::enqueue m=%d, n/4=%d, b=%d\n", m, qn, batchSize);

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

  return 0;
}
