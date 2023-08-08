/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12
 * Use the onnx operator "DecodeNMS" for operator decode & nms

 */

#pragma once

#include "plugin.hpp"
#include "serialize.hpp"

#include <cassert>

using namespace nvinfer1;

extern int TRT_DEBUGLEVEL;  // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#define DPRINTF(level, x...)         \
  do {                               \
    if ((level) <= TRT_DEBUGLEVEL) { \
      printf(x);                     \
    }                                \
  } while (0)

#define FPNNUM (5 * 2)
class DecodeAndNMSPlugin final : public onnx2trt::Plugin {
  bool _initialized = {false};

  int _detections_per_im = 100;
  float _nms_thresh = 0.5f;
  size_t _count = 0;
  float score_thresh = 0.3f;  // 0.05f;
  int top_n = 1000;
  std::vector<int> _scales = {8, 16, 32, 64, 128};
  const int decodeBufsize = 46137856;

 protected:
  void deserialize(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DPRINTF(2, "DecodeAndNMSPlugin serialLength=%zu\n", serialLength);
    if (serialLength >= 12) {
      deserialize_value(&serialData, &serialLength, &_nms_thresh);
      deserialize_value(&serialData, &serialLength, &_detections_per_im);
      deserialize_value(&serialData, &serialLength, &_scales);
    }
  }

  size_t getSerializationSize() override {
    return getBaseSerializationSize() + serialized_size(_nms_thresh) +
           serialized_size(_detections_per_im) + serialized_size(_scales);
  }

  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _nms_thresh);
    serialize_value(&buffer, _detections_per_im);
    serialize_value(&buffer, _scales);
  }

 public:
  DecodeAndNMSPlugin() {}
  DecodeAndNMSPlugin(void const *serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
    DPRINTF(2, "DecodeAndNMSPlugin deserialize(%d,%zu,%f)\n",
            _detections_per_im, _scales.size(), _nms_thresh);
  }
  DecodeAndNMSPlugin(int detections_per_im, std::vector<int> &scales,
                     float nms_thresh)
      : _detections_per_im(detections_per_im),
        _nms_thresh(nms_thresh),
        _scales(scales) {
    DPRINTF(1, "DecodeAndNMSPlugin create (%d,%zu,%f)\n", _detections_per_im,
            _scales.size(), _nms_thresh);
  }
  ~DecodeAndNMSPlugin() { terminate(); }
  virtual const char *getPluginType() const override { return "DecodeAndNMS"; }
  virtual int getNbOutputs() const override { return 3; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputDims,
                                             int nbInputDims) override {
    // assert(nbInputs == 10);
    assert(index < FPNNUM);
    nvinfer1::Dims const &input = inputDims[0];
    nvinfer1::Dims output = Dims3(_detections_per_im, (index == 1 ? 4 : 1), 1);
    ;

    DPRINTF(1, "Input shape=%d(%d,%d,%d)\n", input.nbDims, input.d[0],
            input.d[1], input.d[2]);
    DPRINTF(1, "Output shape=%d(%d,%d,%d)\n", output.nbDims, output.d[0],
            output.d[1], output.d[2]);
    return output;
  }
  int initialize() override;  // TODO()
  void terminate() override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return ((type == nvinfer1::DataType::kFLOAT));
  }
  size_t getWorkspaceSize(int maxBatchSize) const override {
    int nsize = decodeBufsize + top_n * 6 * 5 * sizeof(float);
    DPRINTF(2, "DecodeAndNMSPlugin getWorkspaceSize = %d", nsize);
    return nsize;
  }
};
