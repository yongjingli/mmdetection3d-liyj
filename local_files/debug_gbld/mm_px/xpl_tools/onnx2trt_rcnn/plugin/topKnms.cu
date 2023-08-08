#include "topKnms.h"

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>

#define CUDA_CHECK(condition)                               \
  /* Code block avoids redefinition of cudaError_t error */ \
  do {                                                      \
    cudaError_t error = condition;                          \
    if (error != cudaSuccess) {                             \
      std::cout << cudaGetErrorString(error) << std::endl;  \
    }                                                       \
  } while (0)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const *const a, float const *const b) {
  float left = max(a[1], b[1]), right = min(a[3], b[3]);
  float top = max(a[2], b[2]), bottom = min(a[4], b[4]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  return interS / (a[0] + b[0] - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes,
                           unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  const int n_boxdim = 5;

  // if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * n_boxdim];
  if (threadIdx.x < col_size) {
    int index = (threadsPerBlock * col_start + threadIdx.x) * n_boxdim;
    block_boxes[threadIdx.x * n_boxdim + 0] = dev_boxes[index + 0];
    block_boxes[threadIdx.x * n_boxdim + 1] = dev_boxes[index + 1];
    block_boxes[threadIdx.x * n_boxdim + 2] = dev_boxes[index + 2];
    block_boxes[threadIdx.x * n_boxdim + 3] = dev_boxes[index + 3];
    block_boxes[threadIdx.x * n_boxdim + 4] = dev_boxes[index + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * n_boxdim;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * n_boxdim) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void nms(int *keep_out, int *num_out, const float *boxes_dev, int boxes_num,
         int boxes_dim, float nms_overlap_thresh, int *order,
         cudaStream_t stream) {
  static unsigned long long *mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  if (NULL == mask_dev) {
    CUDA_CHECK(cudaMalloc(&mask_dev,
                          boxes_num * col_blocks * sizeof(unsigned long long)));
  }

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads, 0, stream>>>(boxes_num, nms_overlap_thresh,
                                             boxes_dev, mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(
      cudaMemcpyAsync(&mask_host[0], mask_dev,
                      sizeof(unsigned long long) * boxes_num * col_blocks,
                      cudaMemcpyDeviceToHost, stream));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  cudaStreamSynchronize(stream);
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  // CUDA_CHECK(cudaFree(boxes_dev));
  // CUDA_CHECK(cudaFree(mask_dev));
}

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 64/256/512 threads per block( xavier 8 SMs x 64 ALU )
const int CAFFE_CUDA_NUM_THREADS = 256;
// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

struct RECT {
  float left;
  float top;
  float right;
  float bottom;
};

__constant__ int g_ACNUM;
__constant__ float g_ANCHORS[5 * 8 * 4];

void SetCudaSymbol(int acNum, float *anchorBuf, int anNum) {
  cudaMemcpyToSymbol(g_ACNUM, &acNum, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(g_ANCHORS, anchorBuf, sizeof(float) * anNum, 0,
                     cudaMemcpyHostToDevice);
}

__global__ void GetShiftedAnchors_kernel(const int nthreads, int im_height,
                                         int im_width, int sc_height,
                                         int sc_width, int PM, int _feat_stride,
                                         int *order, float *bbox_deltas_f,
                                         float *im_i_boxes_f, bool clip = true) {
  CUDA_1D_KERNEL_LOOP(oi, nthreads) {
    // 1. Generate proposals from bbox deltas and shifted anchors
    int index = order[oi];
    int imgsize = sc_height * sc_width;
    int x = index % sc_width;
    int y = (index / sc_width) % sc_height;
    int acid = (index / imgsize);
    float shift_x = x * _feat_stride;
    float shift_y = y * _feat_stride;
    // const float4 *ac = cANCHORS[PM][acid];
    const float *ac = &g_ANCHORS[(PM * g_ACNUM + acid) * 4];
    // float min_size = 0;  // cfg[cfg_key].RPN_MIN_SIZE
    // min_size*= im_info.f[2];
    RECT newanch = {ac[0] + shift_x, ac[1] + shift_y, ac[2] + shift_x,
                    ac[3] + shift_y};

    // Transform anchors into proposals via bbox transformations
    // onebbox_transform
    float width = newanch.right - newanch.left + 1.0f;
    float height = newanch.bottom - newanch.top + 1.0f;
    float ctr_x = newanch.left + 0.5f * width;
    float ctr_y = newanch.top + 0.5f * height;
    int id3 = index % imgsize;
    float dx = bbox_deltas_f[acid * 4 * imgsize + id3];
    float dy = bbox_deltas_f[(acid * 4 + 1) * imgsize + id3];
    float dw = bbox_deltas_f[(acid * 4 + 2) * imgsize + id3];
    float dh = bbox_deltas_f[(acid * 4 + 3) * imgsize + id3];
    dw = min(dw, 4.135166556742356f);
    dh = min(dh, 4.135166556742356f);
    float pred_ctr_x = dx * width + ctr_x;
    float pred_ctr_y = dy * height + ctr_y;
    float pred_w = exp(dw) * width;
    float pred_h = exp(dh) * height;
    float left = pred_ctr_x - 0.5f * pred_w;
    float top = pred_ctr_y - 0.5f * pred_h;
    float right = pred_ctr_x + 0.5f * pred_w - 1;
    float bottom = pred_ctr_y + 0.5f * pred_h - 1;

    // 2. clip proposals to image (may result in proposals with zero area
    if (clip) { // fsd model not clip, add by dyg
      if (left < 0) left = 0;
      if (left > im_width - 1) left = im_width - 1;

      if (right < 0) right = 0;
      if (right > im_width - 1) right = im_width - 1;

      if (top < 0) top = 0;
      if (top > im_height - 1) top = im_height - 1;
      if (bottom < 0) bottom = 0;
      if (bottom > im_height - 1) bottom = im_height - 1;
    }

    // 3. remove predicted boxes with either height or width < min_size
    // if( (right - left +1)>=min_size && (bottom - top+1)>=min_size )
    {  // min_size = 0 , no roi may removed
      int offset = oi * 5;
      im_i_boxes_f[offset] = (right - left + 1) * (bottom - top + 1);
      im_i_boxes_f[offset + 1] = left;
      im_i_boxes_f[offset + 2] = top;
      im_i_boxes_f[offset + 3] = right;
      im_i_boxes_f[offset + 4] = bottom;
    }  // else order[ oi ] = -1;
#if 0
    if (oi < 0) {
      int i = index;
      printf("boxes[%d] = %d {%f,%f,%f,%f}\n", i, id3 * 3 + acid, newanch.left,
             newanch.top, newanch.right, newanch.bottom);
      printf("deltas[%d] = {%f,%f,%f,%f}\n", i, dx, dy, dw, dh);
      printf("pred_boxes[%d] = {%f,%f,%f,%f}\n", oi, left, top, right, bottom);
    }
#endif
  }
}

struct is_sign {
  __host__ __device__ bool operator()(int &x) { return (x >= 0); }
};

int GetShiftedAnchors(int threadn, int im_height, int im_width, int sc_height,
                      int sc_width, int PM, int feat_stride, int *order,
                      float *bbox_deltas_f, float *im_i_boxes_f,
                      cudaStream_t stream, bool clip) {
  GetShiftedAnchors_kernel<<<CAFFE_GET_BLOCKS(threadn), CAFFE_CUDA_NUM_THREADS,
                             0, stream>>>(threadn, im_height, im_width,
                                          sc_height, sc_width, PM, feat_stride,
                                          order, bbox_deltas_f, im_i_boxes_f, clip);

  return threadn;
}

int sortByKey(void *d_temp_storage, size_t &temp_storage_bytes,
              const float *d_keys_in, float *d_keys_out, const int *d_values_in,
              int *d_values_out, int num_items, cudaStream_t stream) {
  thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, 0, sizeof(*d_keys_in) * 8, stream);
  return 0;
}
