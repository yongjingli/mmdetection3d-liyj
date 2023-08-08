/* For operator RoIAlign(CUDA,FP32), BatchPermutation(CUDA,FP32/FP16) of
 * maskrcnn (rpn&fpn) Copyright (c) 2018, Xiaopeng. All rights reserved. Create
 * by caizw @ 2018.9.12
 */

#include <cuda_fp16.h>
#include <cassert>
#include "MaskrcnnPlugin.hpp"

#define CHECK_CUDA(call)         \
  do {                           \
    cudaError_t status = call;   \
    if (status != cudaSuccess) { \
      return status;             \
    }                            \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

template <typename T>
__device__ T bilinear_interpolate(const T *bottom_data, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void RoIAlignForward(const int nthreads, const T *bottom_data,
                                const T spatial_scale, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const int sampling_ratio,  // ==2
                                const T *bottom_rois,
                                int roi_cols,  //==5
                                T *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // RoI could have 4 or 5 columns
    const T *offset_bottom_rois = bottom_rois + n * roi_cols + 1;

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T *offset_bottom_data = bottom_data + c * height * width;

#define FIXED_SAMPLINGRATIO2 \
  1  // fixed sampling_ratio==2, latency 14.37ms -> 11.16ms
#ifndef FIXED_SAMPLINGRATIO2
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x,
                                     index);
        output_val += val;
      }
    }
    output_val /= count;
    top_data[index] = output_val;
#else
    // We use roi_bin_grid to sample the grid and mimic integral
    const int roi_bin_grid_h = 2;  // e.g., = 2
    const int roi_bin_grid_w = 2;

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
    const T y0 =
        roi_start_h + ph * bin_size_h +
        static_cast<T>(0.5f / roi_bin_grid_h) * bin_size_h;  // e.g., 0.5, 1.5
    const T y1 = y0 + 0.5f * bin_size_h;                     // e.g., 0.5, 1.5
    const T x0 = roi_start_w + pw * bin_size_w +
                 static_cast<T>(0.5f / roi_bin_grid_w) * bin_size_w;
    const T x1 = roi_start_w + pw * bin_size_w +
                 static_cast<T>(1.5f / roi_bin_grid_w) * bin_size_w;
    top_data[index] = (bilinear_interpolate(offset_bottom_data, height, width,
                                            y0, x0, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y0, x1, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y1, x0, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y1, x1, index)) /
                      count;
#endif

    // if( 40 == height && 0==(pw+ph+c)) printf("ROIAlign %d v=%f\n",index,
    // top_data[index] );
  }
}

int RoiAlignForward(nvinfer1::DataType dataType, const void *const *inputs,
                    void **outputs, nvinfer1::Dims input_dims,
                    nvinfer1::Dims roi_dims, nvinfer1::Dims output_dims,
                    float spatial_scale_, int sampling_ratio_, void *workspace,
                    cudaStream_t stream) {
  int output_size =
      output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * output_dims.d[3];
  DPRINTF(2,
          "RoiAlignForward fp%d out{%d,%d,%d,%d} spatial_scale_=%f "
          "sampling_ratio_=%d\n",
          ((dataType == nvinfer1::DataType::kFLOAT) ? 32 : 16),
          output_dims.d[0], output_dims.d[1], output_dims.d[2],
          output_dims.d[3], spatial_scale_, sampling_ratio_);

  {
    RoIAlignForward<<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS, 0,
                      stream>>>(
        output_size, (const float *)inputs[0], spatial_scale_, input_dims.d[0],
        input_dims.d[1], input_dims.d[2], output_dims.d[2], output_dims.d[3],
        sampling_ratio_, (const float *)inputs[1], roi_dims.d[1],
        (float *)outputs[0]);
  }

  return 0;
}

// fixed sampling_ratio==2,
__global__ void RoiAlignWithFeatID(const int nthreads,
                                   const float *bottom_datas[6],
                                   const int channels, const int im_h,
                                   const int im_w, const int pooled_height,
                                   const int pooled_width,
                                   const float *bottom_rois,
                                   int roi_cols,  //==5
                                   float *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // RoI could have 5 columns
    const float *offset_bottom_rois = bottom_rois + n * roi_cols;

    // Do not using rounding; this implementation detail is critical
    const int level_idx = (int)offset_bottom_rois[0];  // fpn level 2~5
    const int feat_div = (1 << level_idx);             // 4,8,16,32
    const float spatial_scale = 1.0f / feat_div;       //
    const int height = im_h / feat_div;
    const int width = im_w / feat_div;

    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = max(roi_end_w - roi_start_w, (float)1.);
    float roi_height = max(roi_end_h - roi_start_h, (float)1.);
    float bin_size_h = static_cast<float>(roi_height) / pooled_height;
    float bin_size_w = static_cast<float>(roi_width) / pooled_width;

    const float *offset_bottom_data =
        bottom_datas[level_idx] + c * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    const int roi_bin_grid_h = 2;  // e.g., = 2
    const int roi_bin_grid_w = 2;

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
    const float y0 = roi_start_h + ph * bin_size_h +
                     (0.5f / roi_bin_grid_h) * bin_size_h;  // e.g., 0.5, 1.5
    const float y1 = y0 + 0.5f * bin_size_h;                // e.g., 0.5, 1.5
    const float x0 =
        roi_start_w + pw * bin_size_w + (0.5f / roi_bin_grid_w) * bin_size_w;
    const float x1 =
        roi_start_w + pw * bin_size_w + (1.5f / roi_bin_grid_w) * bin_size_w;
    top_data[index] = (bilinear_interpolate(offset_bottom_data, height, width,
                                            y0, x0, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y0, x1, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y1, x0, index) +
                       bilinear_interpolate(offset_bottom_data, height, width,
                                            y1, x1, index)) /
                      count;
  }
}

#define CUDA_ERROR_CHECK
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
#endif

  return;
}

int RoiAlignWithFeatID(const void *const *featPtr, void **outputs,
                       int input_dims[], int roi_cols, int output_dims[],
                       void *workspace, cudaStream_t stream) {
  int output_size =
      output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
  DPRINTF(2, "RoiAlignWithFeatID fp32 out{%d,%d,%d,%d}\n", output_dims[0],
          output_dims[1], output_dims[2],
          output_dims[3]);  // 100,256,7,7

  {
    CHECK_CUDA(cudaMemcpyAsync(workspace, featPtr, sizeof(*featPtr) * 6,
                               cudaMemcpyHostToDevice, stream));

    RoiAlignWithFeatID<<<CAFFE_GET_BLOCKS(output_size), CAFFE_CUDA_NUM_THREADS,
                         0, stream>>>(
        output_size, (const float **)workspace, input_dims[0], input_dims[1],
        input_dims[2], output_dims[2], output_dims[3],
        (const float *)outputs[0], roi_cols, (float *)outputs[1]);
    cudaCheckError();
  }

  return 0;
}

// Plugin Interface:
// enquequ----------------------------------------------------------
extern int roialignsize[4 * 16];
int RoIAlignPlugin::enqueue(int batchSize, const void *const *inputs,
                            void **outputs, void *workspace,
                            cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip RoIAlignPlugin::enqueue!!\n");
    return 0;
  }

  auto const &input_dims = this->getInputDims(0);
  auto const &roi_dims = this->getInputDims(1);
  float spatial_scale_ = _scale[0];
  int pooled_height_ = _poolsize[0];
  int pooled_width_ = _poolsize[1];
  int sampling_ratio_ = _poolsize[2];

#ifndef FIXED_SAMPLINGRATIO2
  assert(sampling_ratio_ == 0);
#else
  assert(sampling_ratio_ == 2);  // optimized for sampling_ratio_ == 2
#endif

  // Y->Resize(R.dim32(0), X.dim32(1), pooled_height_, pooled_width_);
  nvinfer1::Dims output_dims = input_dims;
  output_dims.nbDims = 4;
  output_dims.type[3] = output_dims.type[2];
  output_dims.d[0] = roi_dims.d[0];
  output_dims.d[1] = input_dims.d[0];  // 256
  output_dims.d[2] = pooled_height_;
  output_dims.d[3] = pooled_width_;

  int lvl = -log2(spatial_scale_);
  for (int nb = 0; nb < batchSize; nb++) {  // prcoess batch size <=16
    output_dims.d[0] = roialignsize[lvl - 2 + 4 * nb];
    DPRINTF(2, "RoIAlignPlugin roi_num = %d \n", output_dims.d[0]);
    if (output_dims.d[0] > 0) {
      const void *cur_inputs[2] = {
          (float *)inputs[0] +
              nb * input_dims.d[0] * input_dims.d[1] * input_dims.d[2],
          (float *)inputs[1] + nb * roi_dims.d[0] * roi_dims.d[1]};
      void *cur_outputs[1] = {(float *)outputs[0] +
                              nb * roi_dims.d[0] * output_dims.d[1] *
                                  output_dims.d[2] * output_dims.d[3]};
      RoiAlignForward(getDataType(), cur_inputs, cur_outputs, input_dims,
                      roi_dims, output_dims, spatial_scale_, sampling_ratio_,
                      workspace, stream);
    }
  }
  // cudaStreamSynchronize(stream);
  DPRINTF(2, "RoIAlignPlugin finished\n");

  return 0;
}

template <typename T>
__global__ void BatchPermutationKernel(int N, int C, int H, int W, const T *src,
                                       const T *indices, T *dst) {
  CUDA_1D_KERNEL_LOOP(index, N * C * H * W) {
    int w = index % W;
    int h = (index / W) % H;
    int c = (index / W / H) % C;
    int n = (index / W / H / C);
    unsigned int idx = ((unsigned int)indices[n]) % (N * 4);  // avoid overflow
    {
      dst[n * C * H * W + c * H * W + h * W + w] =
          src[idx * C * H * W + c * H * W + h * W + w];  // about 4.7ms
      // if( 0 == c+h+w )printf("CUDA %d mv %f,%d -> %d v=%f\n",index,
      // indices[n], idx, n, src[idx * C * H * W + c * H * W + h * W + w]);
    }
  }
}

template <typename T>
__global__ void BatchPermutationKernel2(int N, int C, int HW, int W,
                                        const T *src, const T *indices,
                                        T *dst) {
  CUDA_1D_KERNEL_LOOP(index, N * C) {
    int c = (index) % C;
    int n = (index / C);
    int idx = (int)indices[n];
    {
      memcpy(&dst[n * C * HW + c * HW], &src[idx * C * HW + c * HW],
             HW * sizeof(T));  // about 100ms,very slow
      // if( 0 == c+h+w )printf("CUDA %d mv %f,%d -> %d v=%f\n",index,
      // indices[n], idx, n, src[idx * C * H * W + c * H * W + h * W + w]);
    }
  }
}

int BatchPermutationPlugin::enqueue(int batchSize, const void *const *inputs,
                                    void **outputs, void *workspace,
                                    cudaStream_t stream) {
  if (TRT_DEBUGLEVEL == -1) {
    printf("Skip BatchPermutationPlugin::enqueue!!\n");
    return 0;
  }

  auto const &input_dims = this->getInputDims(0);
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[0] = post_nms_topN;
  if (3 == output_dims.nbDims) {
    output_dims.nbDims = 4;
    output_dims.d[2] = output_dims.d[3] = sqrtf(output_dims.d[2]);
  }

  DPRINTF(2, "BatchPermutation fp%d out %dx{%d,%d,%d,%d}\n ",
          ((getDataType() == nvinfer1::DataType::kFLOAT) ? 32 : 16), batchSize,
          output_dims.d[0], output_dims.d[1], output_dims.d[2],
          output_dims.d[3]);
  int output_size =
      output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * output_dims.d[3];

  for (int nb = 0; nb < batchSize; nb++) {  // prcoess batch size <=16
    if (getDataType() == nvinfer1::DataType::kFLOAT) {
      BatchPermutationKernel<<<CAFFE_GET_BLOCKS(output_size),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          output_dims.d[0], output_dims.d[1], output_dims.d[2],
          output_dims.d[3],
          static_cast<const float *>(inputs[0]) +
              nb * output_size * 4,  // levels nfp lvl0=4 ( 2~5 )
          static_cast<const float *>(inputs[1]) + nb * output_dims.d[0],
          static_cast<float *>(outputs[0]) + nb * output_size);
    } else {
      BatchPermutationKernel<<<CAFFE_GET_BLOCKS(output_size),
                               CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
          output_dims.d[0], output_dims.d[1], output_dims.d[2],
          output_dims.d[3],
          static_cast<const __half *>(inputs[0]) + nb * output_size,
          static_cast<const __half *>(inputs[1]) + nb * output_dims.d[0],
          static_cast<__half *>(outputs[0]) + nb * output_size);
    }
    DPRINTF(2, "BatchPermutationKernel finished\n ");
  }
  return 0;
}

__global__ void float22Half2Vec(float2 *src, half2 *des, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < size / 2; i += stride)
    des[i] = __float22half2_rn(src[i]);
}
__global__ void half22Float2Vec(half2 *src, float2 *des, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < size / 2; i += stride) des[i] = __half22float2(src[i]);
}

/*
float *pCPU : pointer to CPU data, float32
float *pCPU : pointer to GPU data, float32/float16
int size: the size of data
int type: 0: float32, 1: float16
int direction: 0: CPU to GPU, 1: GPU to CPU
*/
int convertCPU_GPU(void *pCPU, void *pGPU, int size, int type, int direction,
                   cudaStream_t stream, void *pBuffer) {
  if (size <= 0) return 0;

  if (0 == type) {
    if (0 == direction) {
      CHECK_CUDA(cudaMemcpyAsync(pGPU, pCPU, size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(pCPU, pGPU, size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    }
  } else {
    if (0 == direction) {
      CHECK_CUDA(cudaMemcpyAsync(pBuffer, pCPU, size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      float22Half2Vec<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS, 0,
                        stream>>>((float2 *)pBuffer, (half2 *)pGPU, size);
    } else {
      half22Float2Vec<<<CAFFE_GET_BLOCKS(size), CAFFE_CUDA_NUM_THREADS, 0,
                        stream>>>((half2 *)pGPU, (float2 *)pBuffer, size);
      CHECK_CUDA(cudaMemcpyAsync(pCPU, pBuffer, size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    }
  }

  if (1 == direction) {
    cudaStreamSynchronize(stream);
  }

  return cudaGetLastError() != cudaSuccess;
}
