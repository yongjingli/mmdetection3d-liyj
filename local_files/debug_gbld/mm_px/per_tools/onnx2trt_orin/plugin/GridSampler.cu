// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cuh
// and
// https://github.com/pytorch/pytorch/blob/ec683299ebabf297a3504c76248d37be830e4342/aten/src/ATen/native/cuda/GridSampler.cu

#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include "common.h"
#include "ResizeBilinear.hpp"
#include "GridSampler.h"

// #define _CUDA_TIME_POINT_

using mmcv::GridSamplerInterpolation;
using mmcv::GridSamplerPadding;
// using mmcv::TensorDesc;

const int MAXTENSORDIMS = 10;

struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

static __forceinline__ __device__
__half operator*(const __half& a, const int& b)
{
    return a * __int2half_rn(b);
}

static __forceinline__ __device__
__half operator/(const __half& a, const int& b)
{
    return a / __int2half_rn(b);
}

static __forceinline__ __device__
__half operator+(const __half& a, const float& b)
{
    return a + __float2half(b);
}

static __forceinline__ __device__
__half operator-(const __half& a, const int& b)
{
    return a - __int2half_rn(b);
}

static __forceinline__ __device__
__half operator-(const int& a, const __half& b)
{
    return __int2half_rn(a) - b;
}

static __forceinline__ __device__
__half operator+=(const __half& a, const __half& b)
{
    return a + b;
}

static __forceinline__ __device__
bool operator>(const __half& a, const int& b)
{
    return a > __int2half_rn (b);
}

static __forceinline__ __device__
bool operator<(const __half& a, const int& b)
{
    return a < __int2half_rn (b);
}

static __forceinline__ __device__
__half min(const __half& a, const half& b)
{
    return __float2half(min(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half max(const __half& a, const half& b)
{
    return __float2half(max(__half2float(a), __half2float(b)));
}

static __forceinline__ __device__
__half fabs(const __half& a)
{
    //TODO return __habs(a); what happened.
    return __float2half(fabs(__half2float(a)));
}

static __forceinline__ __device__
__half floorf(const __half& a)
{
    return hfloor(a);
}

static __forceinline__ __device__
__half roundf(const __half& a)
{
    return hrint(a);
}

static __forceinline__ __device__
__half fmod(const __half& a, const __half& b)
{
  return __float2half(fmodf(__half2float(a), __half2float(b)));
}



// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
static __forceinline__ __device__ scalar_t
grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__ scalar_t clip_coordinates(scalar_t in,
                                                            int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1),
               ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__ scalar_t reflect_coordinates(scalar_t in,
                                                               int twice_low,
                                                               int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = fmod(in, span);
  int flips = static_cast<int>(floorf(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
static __forceinline__ __device__ scalar_t
safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__ scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H,
                                                        int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ bool within_bounds_3d(int d, int h, int w,
                                                        int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
__global__ void grid_sampler_2d_kernel(
    const int nthreads, const scalar_t *input, const scalar_t *grid,
    scalar_t *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, bool align_corners) {
  int C = input_desc.shape[1];
  int inp_H = input_desc.shape[2];
  int inp_W = input_desc.shape[3];
  int out_H = grid_desc.shape[1];
  int out_W = grid_desc.shape[2];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sH = input_desc.stride[2];
  int inp_sW = input_desc.stride[3];
  int grid_sN = grid_desc.stride[0];
  int grid_sH = grid_desc.stride[1];
  int grid_sW = grid_desc.stride[2];
  int grid_sCoor = grid_desc.stride[3];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sH = output_desc.stride[2];
  int out_sW = output_desc.stride[3];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y coordinates from grid
    scalar_t ix = grid[grid_offset];
    scalar_t iy = grid[grid_offset + grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                           align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                           align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(floorf(ix));
      int iy_nw = static_cast<int>(floorf(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix) * (iy_se - iy);
      scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
      scalar_t se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<scalar_t>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(roundf(ix));
      int iy_nearest = static_cast<int>(roundf(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void grid_sampler_3d_kernel(
    const int nthreads, const scalar_t *input, const scalar_t *grid,
    scalar_t *output, TensorDesc input_desc, TensorDesc grid_desc,
    TensorDesc output_desc, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, bool align_corners) {
  int C = input_desc.shape[1];
  int inp_D = input_desc.shape[2];
  int inp_H = input_desc.shape[3];
  int inp_W = input_desc.shape[4];
  int out_D = grid_desc.shape[1];
  int out_H = grid_desc.shape[2];
  int out_W = grid_desc.shape[3];
  int inp_sN = input_desc.stride[0];
  int inp_sC = input_desc.stride[1];
  int inp_sD = input_desc.stride[2];
  int inp_sH = input_desc.stride[3];
  int inp_sW = input_desc.stride[4];
  int grid_sN = grid_desc.stride[0];
  int grid_sD = grid_desc.stride[1];
  int grid_sH = grid_desc.stride[2];
  int grid_sW = grid_desc.stride[3];
  int grid_sCoor = grid_desc.stride[4];
  int out_sN = output_desc.stride[0];
  int out_sC = output_desc.stride[1];
  int out_sD = output_desc.stride[2];
  int out_sH = output_desc.stride[3];
  int out_sW = output_desc.stride[4];

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int d = (index / (out_H * out_W)) % out_D;
    const int n = index / (out_D * out_H * out_W);
    const int grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    scalar_t ix = grid[grid_offset];
    scalar_t iy = grid[grid_offset + grid_sCoor];
    scalar_t iz = grid[grid_offset + 2 * grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode,
                                           align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode,
                                           align_corners);
    iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode,
                                           align_corners);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      int ix_tnw = static_cast<int>(floorf(ix));
      int iy_tnw = static_cast<int>(floorf(iy));
      int iz_tnw = static_cast<int>(floorf(iz));

      int ix_tne = ix_tnw + 1;
      int iy_tne = iy_tnw;
      int iz_tne = iz_tnw;

      int ix_tsw = ix_tnw;
      int iy_tsw = iy_tnw + 1;
      int iz_tsw = iz_tnw;

      int ix_tse = ix_tnw + 1;
      int iy_tse = iy_tnw + 1;
      int iz_tse = iz_tnw;

      int ix_bnw = ix_tnw;
      int iy_bnw = iy_tnw;
      int iz_bnw = iz_tnw + 1;

      int ix_bne = ix_tnw + 1;
      int iy_bne = iy_tnw;
      int iz_bne = iz_tnw + 1;

      int ix_bsw = ix_tnw;
      int iy_bsw = iy_tnw + 1;
      int iz_bsw = iz_tnw + 1;

      int ix_bse = ix_tnw + 1;
      int iy_bse = iy_tnw + 1;
      int iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {

        *out_ptr_NCDHW = static_cast<scalar_t>(0);
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] *
              tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] *
              tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] *
              tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] *
              tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] *
              bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] *
              bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] *
              bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] *
              bse;
        }
      }
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(roundf(ix));
      int iy_nearest = static_cast<int>(roundf(iy));
      int iz_nearest = static_cast<int>(roundf(iz));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (int c = 0; c < C;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H,
                             inp_W)) {
          *out_ptr_NCDHW =
              inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH +
                         ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

void create_desc(const int *dims, int nb_dims, TensorDesc &desc) {
  memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
  desc.stride[nb_dims - 1] = 1;
  for (int i = nb_dims - 2; i >= 0; --i) {
    desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
  }
}

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int *output_dims,
                 int *input_dims, int *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding,
                 bool align_corners, cudaStream_t stream) {
  TensorDesc input_desc;
  create_desc(input_dims, nb_dims, input_desc);

  TensorDesc output_desc;
  create_desc(output_dims, nb_dims, output_desc);

  TensorDesc grid_desc;
  create_desc(grid_dims, nb_dims, grid_desc);

  int count = 1;
  for (int i = 0; i < nb_dims; ++i) {
    if (i == 1) {
      continue;
    }
    count *= output_desc.shape[i];
  }

  if (nb_dims == 4) {
    grid_sampler_2d_kernel<T>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else if (nb_dims == 5) {
    grid_sampler_3d_kernel<T>
        <<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
            count, input, grid, output, input_desc, grid_desc, output_desc,
            interp, padding, align_corners);
  } else {
    printf("input and grid dims should be 4 or 5\n");
  }
}

void grid_sample_float(float *output, const float *input, const float *grid,
                       int *output_dims, int *input_dims, int *grid_dims,
                       int nb_dims, GridSamplerInterpolation interp,
                       GridSamplerPadding padding, bool align_corners,
                       cudaStream_t stream) {
  grid_sample<float>(output, input, grid, output_dims, input_dims, grid_dims,
                     nb_dims, interp, padding, align_corners, stream);
}

void grid_sample_float16(__half *output, const __half *input, const __half *grid,
        int *output_dims, int *input_dims, int *grid_dims,
        int nb_dims, GridSamplerInterpolation interp,
        GridSamplerPadding padding, bool align_corners,
        cudaStream_t stream) {
    grid_sample<__half>(output, input, grid, output_dims, input_dims, grid_dims,
    nb_dims, interp, padding, align_corners, stream);
}

#include <assert.h>
#include <stdio.h>
#include <chrono>

using mmcv::GridSamplerInterpolation;
using mmcv::GridSamplerPadding;

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridSample"};
}  // namespace

nvinfer1::PluginFieldCollection GridSamplerDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> GridSamplerDynamicCreator::mPluginAttributes;

GridSamplerDynamic::GridSamplerDynamic(const std::string &name, int mode,
                                       int paddingMode, bool alignCorners)
    : mLayerName(name),
      mMode(mode),
      mPaddingMode(paddingMode),
      mAlignCorners(alignCorners) {}

GridSamplerDynamic::GridSamplerDynamic(const std::string name, const void *data,
                                       size_t length) 
    : mLayerName(name) {
  deserialize_value(&data, &length, &mMode);
  deserialize_value(&data, &length, &mPaddingMode);
  deserialize_value(&data, &length, &mAlignCorners);
}

nvinfer1::IPluginV2DynamicExt *GridSamplerDynamic::clone() const noexcept {
  GridSamplerDynamic *plugin =
      new GridSamplerDynamic(mLayerName, mMode, mPaddingMode, mAlignCorners);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GridSamplerDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  for (int i = 2; i < ret.nbDims; ++i) {
    ret.d[i] = inputs[1].d[i - 1];
  }
  return ret;
}

bool GridSamplerDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) noexcept {
  if (pos == 0) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void GridSamplerDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) noexcept {
  // Validate input arguments
}

size_t GridSamplerDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

int GridSamplerDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc,
                                const void *const *inputs, void *const *outputs,
                                void *workSpace, cudaStream_t stream) noexcept {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  nvinfer1::Dims grid_dims = inputDesc[1].dims;
  nvinfer1::Dims output_dims = outputDesc[0].dims;

  using mmcv::GridSamplerInterpolation;
  using mmcv::GridSamplerPadding;

  GridSamplerInterpolation interp_mode = GridSamplerInterpolation::Bilinear;
  switch (mMode) {
    case 0:
      interp_mode = GridSamplerInterpolation::Bilinear;
      break;
    case 1:
      interp_mode = GridSamplerInterpolation::Nearest;
      break;
    default:
      break;
  }

  GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;
  switch (mPaddingMode) {
    case 0:
      padding_mode = GridSamplerPadding::Zeros;
      break;

    case 1:
      padding_mode = GridSamplerPadding::Border;
      break;

    case 2:
      padding_mode = GridSamplerPadding::Reflection;
      break;
    default:
      break;
  }

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      grid_sample_float(
          (float *)outputs[0], (float *)inputs[0], (float *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, interp_mode, padding_mode, mAlignCorners, stream);
      break;
    case nvinfer1::DataType::kHALF:
      grid_sample_float16(
          (__half *)outputs[0], (__half *)inputs[0], (__half *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, interp_mode, padding_mode, mAlignCorners, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType GridSamplerDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GridSamplerDynamic::getPluginType() const noexcept { return PLUGIN_NAME; }

const char *GridSamplerDynamic::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int GridSamplerDynamic::getNbOutputs() const noexcept { return 1; }

int GridSamplerDynamic::initialize() noexcept { return 0; }

void GridSamplerDynamic::terminate() noexcept {}

size_t GridSamplerDynamic::getSerializationSize() const noexcept {
  return sizeof(mMode) + sizeof(mPaddingMode) + sizeof(mAlignCorners);
}

void GridSamplerDynamic::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, mMode);
  serialize_value(&buffer, mPaddingMode);
  serialize_value(&buffer, mAlignCorners);
}

void GridSamplerDynamic::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void GridSamplerDynamic::setPluginNamespace(const char *libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char *GridSamplerDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

GridSamplerDynamicCreator::GridSamplerDynamicCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("interpolation_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GridSamplerDynamicCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char *GridSamplerDynamicCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
GridSamplerDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2 *GridSamplerDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  int mode = 0;
  int paddingMode = 0;
  bool alignCorners = false;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("interpolation_mode") == 0) {
      mode = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("padding_mode") == 0) {
      paddingMode = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("align_corners") == 0) {
      alignCorners = (bool)(static_cast<const int *>(fc->fields[i].data)[0]);
    }
  }
  DPRINTF(1, "GridSampler interpolation_mode: %d padding_mode: %d align_corners: %d\n", mode, paddingMode, alignCorners);
  GridSamplerDynamic *plugin =
      new GridSamplerDynamic(name, mode, paddingMode, alignCorners);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GridSamplerDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new GridSamplerDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void GridSamplerDynamicCreator::setPluginNamespace(const char *libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char *GridSamplerDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}
