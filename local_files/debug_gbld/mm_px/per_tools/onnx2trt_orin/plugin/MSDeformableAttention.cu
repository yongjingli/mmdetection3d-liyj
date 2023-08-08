
#include <cuda_fp16.h>
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include "common.h"
#include "MSDeformableAttention.h"
#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <string.h>

#define DEBUG_K_PRINT

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_interpolation(const scalar_t* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const float &h, const float &w, const int &m, const int &c)
{
  const float h_low_f = floorf(h);
  const float w_low_f = floorf(w);
  const int h_low = static_cast<int>(h_low_f);
  const int w_low = static_cast<int>(w_low_f);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h - h_low_f;
  const float lw = w - w_low_f;
  const float hh = 1.0f - lh;
  const float hw = 1.0f - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = static_cast<scalar_t>(hh * hw);
  const scalar_t w2 = static_cast<scalar_t>(hh * lw); 
  const scalar_t w3 = static_cast<scalar_t>(lh * hw); 
  const scalar_t w4 = static_cast<scalar_t>(lh * lw);
  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t * data_value, 
                                                const int * data_spatial_shapes,
                                                const int * data_level_start_index, 
                                                const float *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t * data_col)
{
  CUDA_1D_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels; // (0, 32)
    _temp /= channels; 
    const int sampling_index = _temp; // (0, 1x1600x8)
    const int m_col = _temp % num_heads; // (0, 8)
    _temp /= num_heads;
    const int q_col = _temp % num_query; // (0, 1600)
    _temp /= num_query;
    const int b_col = _temp; // (0)

    scalar_t *data_col_ptr = data_col + index; // d + (0, 1x1600x8x32)
    int data_weight_ptr = sampling_index * num_levels * num_point; // (0, 1x1600x8x20x4) stride 80=20x4
    int data_loc_w_ptr = data_weight_ptr << 1; // (0, 1x1600x8x20x4x2) stride 160
    const int qid_stride = num_heads * channels; // 256=32x8
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride; // 0x127500x32x8
    scalar_t col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col) // (0, 20)
    {

      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col * 2;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const float loc_w = data_sampling_loc[data_loc_w_ptr];
        const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const float h_im = loc_h * spatial_h - 0.5;
        const float w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_interpolation(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(const scalar_t * data_value,
                              const int * data_spatial_shapes, 
                              const int * data_level_start_index, 
                              const float * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              scalar_t * data_col,
                              cudaStream_t stream)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = THREADS_PER_BLOCK;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}


namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MSDeformableAttention"};
}  // namespace

nvinfer1::PluginFieldCollection MSDeformableAttentionCreator::mFC{};
std::vector<nvinfer1::PluginField> MSDeformableAttentionCreator::mPluginAttributes;

MSDeformableAttention::MSDeformableAttention(const std::string &name, 
        const int* spatial_shapes, int spatial_shapes_length, int mode,
        int paddingMode, bool alignCorners)
    : mLayerName(name),
      mMode(mode),
      mPaddingMode(paddingMode),
      mAlignCorners(alignCorners) {
  if (spatial_shapes_length > 0) {
    mSpatialShapes.resize(spatial_shapes_length);
    mSpatialShapesLength = spatial_shapes_length >> 1;
    mLevelStartIndex.resize(mSpatialShapesLength);
    for (int j=0; j<spatial_shapes_length; ++j) {
      mSpatialShapes[j] = spatial_shapes[j];
    }
    int start_index = 0;
    for (int j=0; j<mSpatialShapesLength; ++j) {
      mLevelStartIndex[j] = start_index;
      start_index += mSpatialShapes[2*j] * mSpatialShapes[2*j+1];
    }
  }
}

MSDeformableAttention::MSDeformableAttention(const std::string name, const void *data,
                                       size_t length) 
    : mLayerName(name) {
  const char *d = static_cast<const char *>(data);
  const char *const a = d;
  mSpatialShapesLength = read<int>(d);
  mSpatialShapes.resize(mSpatialShapesLength*2);
  mLevelStartIndex.resize(mSpatialShapesLength);
  int * p = mSpatialShapes.data();
  read<int>(d, p, mSpatialShapesLength*2);
  p = mLevelStartIndex.data();
  read<int>(d, p, mSpatialShapesLength);
  mMode = read<int>(d);
  mPaddingMode = read<int>(d);
  mAlignCorners = read<int>(d);
  if (d != a + length) {
    DPRINTF(1, "MSDeformableAttention init error! offset = %zu  length = %zu\n", (size_t)(d-a), length);
  }

}

nvinfer1::IPluginV2DynamicExt *MSDeformableAttention::clone() const noexcept {
  const int * shapes = mSpatialShapes.data();
  MSDeformableAttention *plugin =
      new MSDeformableAttention(mLayerName, shapes, 
                                mSpatialShapesLength*2, mMode, mPaddingMode, mAlignCorners);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs MSDeformableAttention::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims - 1;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[1].d[1];
  ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);
  return ret;
}

bool MSDeformableAttention::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) noexcept {
    
  if (pos == 0) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos == 1) {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

void MSDeformableAttention::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) noexcept {
  // Validate input arguments
  CHECK_CUDA(cudaMalloc(&mSpatialShapes_d, sizeof(int) * mSpatialShapes.size()));
  CHECK_CUDA(cudaMalloc(&mLevelStartIndex_d, sizeof(int) * mLevelStartIndex.size()));
  CHECK_CUDA(cudaMemcpy(mSpatialShapes_d, mSpatialShapes.data(), 
    sizeof(int) * mSpatialShapes.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(mLevelStartIndex_d, mLevelStartIndex.data(), 
    sizeof(int) * mLevelStartIndex.size(), cudaMemcpyHostToDevice));
  if (mSpatialShapes_d == nullptr)
    printf("mSpatialShapes_d malloc error\n");
  if (mLevelStartIndex_d == nullptr)
    printf("mLevelStartIndex_d malloc error\n");
}

size_t MSDeformableAttention::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

int MSDeformableAttention::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                const nvinfer1::PluginTensorDesc *outputDesc,
                                const void *const *inputs, void *const *outputs,
                                void *workSpace, cudaStream_t stream) noexcept {
  // value
  const int batch = inputDesc[0].dims.d[0];
  const int spatial_size = inputDesc[0].dims.d[1];
  const int num_heads = inputDesc[0].dims.d[2];
  const int channels = inputDesc[0].dims.d[3];

  const int num_levels = inputDesc[1].dims.d[3];

  // spatial_shape
  const int num_query = inputDesc[1].dims.d[1];
  const int num_point = inputDesc[1].dims.d[4];

  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto per_output_size = num_query * num_heads * channels;
  // auto per_output_size = outputDesc[1].dims.d[0] * outputDesc[1].dims.d[1] * outputDesc[1].dims.d[2];

  auto data_type = inputDesc[0].type;
  for (int b=0; b<batch; ++b) {
    
    switch (data_type) {
      case nvinfer1::DataType::kFLOAT:
        ms_deformable_im2col_cuda<float>(static_cast<const float *>(inputs[0]) + b * per_value_size,
              (const int *)mSpatialShapes_d, 
              (const int *)mLevelStartIndex_d, 
              static_cast<const float *>(inputs[1]) + b * per_sample_loc_size,
              static_cast<const float *>(inputs[2]) + b * per_attn_weight_size,
              batch,
              spatial_size, 
              num_heads, 
              channels, 
              num_levels, 
              num_query,
              num_point,
              static_cast<float *>(outputs[0]) + b * per_output_size,
              stream);
        break;
      case nvinfer1::DataType::kHALF:
        ms_deformable_im2col_cuda<__half>(static_cast<const __half *>(inputs[0]) + b * per_value_size,
              (const int *)mSpatialShapes_d, 
              (const int *)mLevelStartIndex_d, 
              static_cast<const float *>(inputs[1]) + b * per_sample_loc_size,
              static_cast<const __half *>(inputs[2]) + b * per_attn_weight_size,
              batch,
              spatial_size, 
              num_heads, 
              channels, 
              num_levels, 
              num_query,
              num_point,
              static_cast<__half *>(outputs[0]) + b * per_output_size,
              stream);
        break;
      default:
        return 1;
        break;
    }
  }
  return 0;
}

nvinfer1::DataType MSDeformableAttention::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *MSDeformableAttention::getPluginType() const noexcept { return PLUGIN_NAME; }

const char *MSDeformableAttention::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int MSDeformableAttention::getNbOutputs() const noexcept { return 1; }

int MSDeformableAttention::initialize() noexcept { 
  return 0; 
}

void MSDeformableAttention::terminate() noexcept {
  if (mSpatialShapes_d != nullptr) {
    cudaFree(mSpatialShapes_d);
    mSpatialShapes_d = nullptr;
  }
  if (mLevelStartIndex_d != nullptr) {
    cudaFree(mLevelStartIndex_d);
    mLevelStartIndex_d = nullptr;
  }
}

size_t MSDeformableAttention::getSerializationSize() const noexcept {
  return sizeof(mSpatialShapesLength) + 
    sizeof(int) * mSpatialShapes.size() + 
    sizeof(int) * mLevelStartIndex.size() +
    sizeof(mMode) + 
    sizeof(mPaddingMode) + 
    sizeof(mAlignCorners);
}

void MSDeformableAttention::serialize(void *buffer) const noexcept {
  std::string mNamespace;
  char *d = static_cast<char *>(buffer);
  const char *const a = d;
  write(d, mSpatialShapesLength);
  write(d, mSpatialShapes.data(), mSpatialShapes.size());
  write(d, mLevelStartIndex.data(), mLevelStartIndex.size());
  write(d, mMode);
  write(d, mPaddingMode);
  write(d, mAlignCorners);
  if (d != a + getSerializationSize()) {
    DPRINTF(1, "MSDeformableAttention serialize error!  %ld %ld\n",
     (size_t)(d-a), getSerializationSize());    
  }
}

void MSDeformableAttention::destroy() noexcept {
  // This gets called when the network containing plugin is destroyed

  delete this;
}

void MSDeformableAttention::setPluginNamespace(const char *libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char *MSDeformableAttention::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

MSDeformableAttentionCreator::MSDeformableAttentionCreator() {
  mPluginAttributes.clear();
  // mPluginAttributes.emplace_back(nvinfer1::PluginField("spatial_shapes_num"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("spatial_shapes"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("interpolation_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding_mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *MSDeformableAttentionCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char *MSDeformableAttentionCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
MSDeformableAttentionCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2 *MSDeformableAttentionCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  int mode = 0;
  int paddingMode = 0;
  bool alignCorners = false;
  std::vector<int> spatial_shapes;
  int spatial_shapes_length = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);
    
    if (field_name.compare("spatial_shapes") == 0) {
      // if (nvinfer1::PluginFieldType::kINT32 == fc->fields[i].type)
      spatial_shapes.resize(fc->fields[i].length);
      spatial_shapes_length = fc->fields[i].length;
      const int * ptr = static_cast<const int *>(fc->fields[i].data);
      for (int j=0; j<spatial_shapes_length; ++j) {
        spatial_shapes[j] = ptr[j];
      }
    }
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

  MSDeformableAttention *plugin =
      new MSDeformableAttention(name, spatial_shapes.data(), spatial_shapes_length, mode, paddingMode, alignCorners);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *MSDeformableAttentionCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new MSDeformableAttention(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void MSDeformableAttentionCreator::setPluginNamespace(const char *libNamespace) noexcept {
  mNamespace = libNamespace;
}

const char *MSDeformableAttentionCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}
