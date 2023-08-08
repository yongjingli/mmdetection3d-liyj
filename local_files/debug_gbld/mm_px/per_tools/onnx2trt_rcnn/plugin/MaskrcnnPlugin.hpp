/*
 * Copyright (c) 2018, Xiaopeng. All rights reserved.
 * Create by caizw @ 2018.9.4
 * Convert More operators to tensorRT:
 * GenerateProposalsOp(generate_proposals.py),
 CollectAndDistributeFpnRpnProposalsOp(collect_and_distribute_fpn_rpn_proposals.py)
 * RoIAlign(roi_align_op.cu), BatchPermutation(batch_permutation_op.cu),
 ConvTranspose
 * Use the onnx operator "ATen" for operator RoIAlign, BatchPermutation,
 GenerateProposalsOp, CollectAndDistributeFpnRpnProposalsOp
 * caizw@20191112: Support combine all ATen to one plugin
 */

#pragma once

#include <cassert>
#include "RetinaNetPlugin.hpp"
#include "plugin.hpp"
#include "serialize.hpp"

using namespace nvinfer1;

extern int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

// Just for tensorrt5.0.0.x, must be 0 for tensorrt4.x or tensorrt5.0.2.x
#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH <= 1
#define FIX_BUG_TENOSRRT5 1
#endif

// Get mode-dependent configuration
const int PRE_NMS_topN = 512;  // cfg.TEST.RPN_PRE_NMS_TOP_N 1000
extern int pre_nms_topN;
const int POST_NMS_topN = 100;  // cfg.TEST.RPN_POST_NMS_TOP_N 300
extern int post_nms_topN;       // support to change at converting.
const double nms_thresh = 0.7;
// const int num_classes = 6; //cfg.MODEL.NUM_CLASSES. Got from model
const float SCORE_THRESH = 0.7f;    // cfg.TEST.SCORE_THRESH = 0.05
const float NMS = 0.5f;             // cfg.TEST.NMS
const int DETECTIONS_PER_IM = 100;  // cfg.TEST.DETECTIONS_PER_IM

#define COLLECTNUM 6
class CollectAndDisOpPlugin final : public onnx2trt::Plugin {
  bool _initialized = false;
  float *pCPUInput0 = nullptr;
  float *pCPUInput1 = nullptr;
  float *pCPUOutput0 = nullptr;
  float *pCPUOutput1 = nullptr;
  float *pCPUBuffer0 = nullptr;
  float *pCPUBuffer1 = nullptr;

  float *pGPUBuffer0 = nullptr;
  float *pGPUBuffer1 = nullptr;
  int *pGPUIndices = nullptr;

  std::vector<int> anchor_sizes = {32, 64, 128, 256, 512};
  std::vector<int> anchor_stride = {4, 8, 16, 32, 64};
  std::vector<float> anchor_ratios = {0.5f, 1.0f, 2.0f};
  int SZNUM = 5;
  int ACNUM = 3;
  bool hasROIAlign = false;
  int _poolsize[3] = {0, 0, 0};

 protected:
  void deserialize(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);  // 157 Bytes
    DPRINTF(2, "CollectAndDisOp serialLength=%zu\n", serialLength);
    if (serialLength > 32) {
      deserialize_value(&serialData, &serialLength, &anchor_sizes);
      deserialize_value(&serialData, &serialLength, &anchor_stride);
      deserialize_value(&serialData, &serialLength, &anchor_ratios);
    }
    DPRINTF(2, "CollectAndDisOp serialLength=%zu\n", serialLength);
    if (serialLength >= 12) {
      deserialize_value(&serialData, &serialLength, &_poolsize);
      if (2 == _poolsize[2]) hasROIAlign = true;  // fixed samplingRatio = 2
    }
  }

  size_t getSerializationSize() override {
    return getBaseSerializationSize() + serialized_size(anchor_sizes) +
           serialized_size(anchor_stride) + serialized_size(anchor_ratios) +
           (hasROIAlign ? serialized_size(_poolsize) : 0);
  }

  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, anchor_sizes);
    serialize_value(&buffer, anchor_stride);
    serialize_value(&buffer, anchor_ratios);
    if (hasROIAlign) serialize_value(&buffer, _poolsize);
  }

 public:
  CollectAndDisOpPlugin() {}
  CollectAndDisOpPlugin(void const *serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
    DPRINTF(2, "CollectAndDisOp deserialize %zu(%d,%d,%f)\n",
            anchor_ratios.size(), anchor_sizes[0], anchor_stride[0],
            anchor_ratios[0]);
  }
  CollectAndDisOpPlugin(std::vector<int> &an_sizes, std::vector<int> &an_stride,
                        std::vector<float> &an_ratios)
      : anchor_sizes(an_sizes),
        anchor_stride(an_stride),
        anchor_ratios(an_ratios) {
    DPRINTF(1, "CollectAndDisOp create %zu(%d,%d,%f)\n", anchor_ratios.size(),
            anchor_sizes[0], anchor_stride[0], anchor_ratios[0]);
  }
  void SetRoIAlign(int in_pooled_h, int in_pooled_w, int in_samplingRatio) {
    _poolsize[0] = in_pooled_h;
    _poolsize[1] = in_pooled_w;
    _poolsize[2] = in_samplingRatio;
    if (2 == in_samplingRatio) hasROIAlign = true;  // fixed samplingRatio = 2
    DPRINTF(1, "SetROIAlign (%d,%d,%d)\n", _poolsize[0], _poolsize[1],
            _poolsize[2]);
  }
  ~CollectAndDisOpPlugin() { terminate(); }
#if FIX_BUG_TENOSRRT5
  CollectAndDisOpPlugin *clone() const override {
    return new CollectAndDisOpPlugin();
  }
#endif
  virtual const char *getPluginType() const override {
    return "CollectAndDisOp";
  }
  virtual int getNbOutputs() const override {
    return hasROIAlign ? 2 : COLLECTNUM;
  }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputDims) override {
    // assert(nbInputs == 10);
    assert(index < COLLECTNUM);
    nvinfer1::Dims const &input = inputDims[index % 2];
    nvinfer1::Dims output = input;
    if (index < 5) {
      output.d[0] = post_nms_topN;  // 1000
      output.d[1] = 5;
    } else {
      output.d[0] = post_nms_topN;  // 1000
      output.d[1] = 1;
    }

    if (hasROIAlign && 1 == index) {  // Combined with ROIAlign
      output.nbDims = 4;
      output.d[1] = inputDims[2].d[0];
      output.d[2] = _poolsize[0];
      output.d[3] = _poolsize[1];
      DPRINTF(1, "Input[2] shape=%d(%d,%d,%d)\n", inputDims[2].nbDims,
              inputDims[2].d[0], inputDims[2].d[1], inputDims[2].d[2]);
    }

    DPRINTF(1, "Input[%d] shape=%d(%d,%d,%d)\n", index, input.nbDims,
            input.d[0], input.d[1], input.d[2]);
    DPRINTF(1, "Output[%d] shape=%d(%d,%d,%d)\n", index, output.nbDims,
            output.d[0], output.d[1], output.d[2]);
    return output;
  }
  int initialize() override;  // TODO()
  void terminate() override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return (
        (type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW));
  }
  size_t getWorkspaceSize(int maxBatchSize) const override {
    auto const &input0_dims = this->getInputDims(0);
    int nsize = maxBatchSize * input0_dims.d[0] * sizeof(float);
    DPRINTF(2, "CollectAndDisOp getWorkspaceSize = %d", nsize);
    return nsize;
  }
};

class RoIAlignPlugin final : public onnx2trt::Plugin {
  int _poolsize[3];
  float _scale[1];  // spatial_scale
  nvinfer1::Dims _output_dims;

 protected:
  void deserialize(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_poolsize);
    deserialize_value(&serialData, &serialLength, &_scale);
  }
  size_t getSerializationSize() override {
    return serialized_size(_poolsize) + serialized_size(_scale) +
           getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _poolsize);
    serialize_value(&buffer, _scale);
  }

 public:
  RoIAlignPlugin(float const &scale, int const &pool_h, int const &pool_w,
                 int const &sampling_ratio) {
    _scale[0] = scale;
    _poolsize[0] = pool_h;
    _poolsize[1] = pool_w;
    _poolsize[2] = sampling_ratio;
  }
  RoIAlignPlugin(void const *serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
#if FIX_BUG_TENOSRRT5
  RoIAlignPlugin *clone() const override {
    return new RoIAlignPlugin(_scale[0], _poolsize[0], _poolsize[1],
                              _poolsize[2]);
  }
  virtual void terminate() override {}
#endif
  virtual const char *getPluginType() const override { return "RoIAlign"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputDims) override {
    // assert(nbInputs == 2);
    assert(index == 0);
    nvinfer1::Dims const &input = inputDims[0];
    nvinfer1::Dims output = input;
    output.nbDims = 3;
    output.type[3] = output.type[2];
    output.d[0] = inputDims[1].d[0];
    output.d[1] = input.d[0];
    output.d[2] = _poolsize[0] * _poolsize[1];
    output.d[3] = 1;
    DPRINTF(1, "Input shape=%d(%d,%d,%d,%d)\n", input.nbDims, input.d[0],
            input.d[1], input.d[2], input.d[3]);
    DPRINTF(1, "Output shape=%d(%d,%d,%d,%d)\n", output.nbDims, output.d[0],
            output.d[1], output.d[2], output.d[3]);
    return output;
  };
  virtual int initialize() override { return 0; };  // TODO()
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;  // TODO
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return (
        (type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW));
  }
};

#include <math.h>
class BatchPermutationPlugin final : public onnx2trt::Plugin {
  // nvinfer1::Dims _output_dims;
 protected:
  void deserialize(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
  }
  size_t getSerializationSize() override { return getBaseSerializationSize(); }
  void serialize(void *buffer) override { serializeBase(buffer); }

 public:
  BatchPermutationPlugin() {}
  BatchPermutationPlugin(void const *serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
#if FIX_BUG_TENOSRRT5
  BatchPermutationPlugin *clone() const override {
    return new BatchPermutationPlugin();
  }
  virtual void terminate() override {}
#endif
  virtual const char *getPluginType() const override {
    return "BatchPermutation";
  }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputDims) override {
    // assert(nbInputs == 2);
    assert(index == 0);
    nvinfer1::Dims const &input = inputDims[0];
    nvinfer1::Dims output = input;
    output.d[0] = post_nms_topN;  // 1000
    if (3 == output.nbDims) {
      output.nbDims = 4;
      output.d[2] = output.d[3] = sqrtf(output.d[2]);
    }

    DPRINTF(1, "Input shape=%d(%d,%d,%d,%d)\n", input.nbDims, input.d[0],
            input.d[1], input.d[2], input.d[3]);
    DPRINTF(1, "Output shape=%d(%d,%d,%d,%d)\n", output.nbDims, output.d[0],
            output.d[1], output.d[2], output.d[3]);
    return output;
  };
  virtual int initialize() override { return 0; };  // TODO()
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return ((type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kHALF));  //
  }
};

#define KTYPE4 char4  // float4
#define KTYPE char    // float
class GemvInt8Plugin final : public onnx2trt::Plugin {
  int _nrow;
  std::vector<KTYPE> _h_kernel;
  std::vector<float> _h_bias;
  std::vector<float> _h_scale;
  KTYPE *_d_kernel;
  float *_d_bias;
  float *_d_scale;
  bool _initialized = {false};

 protected:
  void deserialize(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_nrow);
    deserialize_value(&serialData, &serialLength, &_h_kernel);
    deserialize_value(&serialData, &serialLength, &_h_bias);
    deserialize_value(&serialData, &serialLength, &_h_scale);
  }
  size_t getSerializationSize() override {
    return (serialized_size(_nrow) + serialized_size(_h_kernel) +
            serialized_size(_h_bias) + serialized_size(_h_scale)) +
           getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _nrow);
    serialize_value(&buffer, _h_kernel);
    serialize_value(&buffer, _h_bias);
    serialize_value(&buffer, _h_scale);
  }

 public:
  GemvInt8Plugin(int rows, nvinfer1::Weights const &kernel,
                 nvinfer1::Weights const &bias);
  GemvInt8Plugin(void const *serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  ~GemvInt8Plugin() { terminate(); }
#if FIX_BUG_TENOSRRT5
  GemvInt8Plugin *clone() const override { return new GemvInt8Plugin(); }
#endif
  virtual const char *getPluginType() const override { return "GemvInt8"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputDims) override {
    assert(index == 0);
    nvinfer1::Dims output = DimsCHW(_nrow, 1, 1);
    output.nbDims = 1;

    DPRINTF(1, "GEMV Input=%d(%d,%d,%d)\n", inputDims[0].nbDims,
            inputDims[0].d[0], inputDims[0].d[1], inputDims[0].d[2]);
    DPRINTF(1, "GEMV Output=%d(%d)\n", output.nbDims, output.d[0]);
    return output;
  };
  int initialize() override;
  void terminate() override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return (type == nvinfer1::DataType::kFLOAT &&
            format == PluginFormat::kNCHW);
  }
  size_t getWorkspaceSize(int maxBatchSize) const override {
    auto const &input0_dims = this->getInputDims(0);
    int nsize = maxBatchSize * input0_dims.d[0] * sizeof(float);
    DPRINTF(2, "GemvInt8Plugin getWorkspaceSize = %d", nsize);
    return nsize;
  }
};

/*
float *pCPU : pointer to CPU data, float32
float *pCPU : pointer to GPU data, float32/float16
int size: the size of data
int type: 0: float32, 1: float16
int direction: 0: CPU to GPU, 1: GPU to CPU
*/
extern "C" {

int convertCPU_GPU(void *pCPU, void *pGPU, int size, int type, int direction,
                   cudaStream_t stream, void *pBuffer);

int RoiAlignForward(nvinfer1::DataType dataType, const void *const *inputs,
                    void **outputs, nvinfer1::Dims input_dims,
                    nvinfer1::Dims roi_dims, nvinfer1::Dims output_dims,
                    float spatial_scale_, int sampling_ratio_, void *workspace,
                    cudaStream_t stream);

int RoiAlignWithFeatID(const void *const *inputs, void **outputs,
                       nvinfer1::Dims input_dims, nvinfer1::Dims roi_dims,
                       nvinfer1::Dims output_dims, float spatial_scale_,
                       int sampling_ratio_, void *workspace,
                       cudaStream_t stream);

void SetImInfoForCollect(float height64, float width64, float scale);

int RunPreMasknet(float **inputDatas, float **outDatas, int *roiNums,
                  int classNum, int width, int height, float im_scale);
}
