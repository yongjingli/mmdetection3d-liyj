/*
 * Copyright (c) 2019, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#include <cuda_fp16.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include "NvInfer.h"

using namespace nvinfer1;

extern "C" int UpsampleV2Forward(int batchSize, float mSale, Dims &mInputDims,
                                 Dims &mOutputDims, DataType &mDataType,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream);
extern "C" int BatchPadConcatForward(int batchSize, Dims &mInputDims,
                                     Dims &mOutputDims, DataType &mInType,
                                     DataType &mOutType, void **weights,
                                     const void *const *inputs, void **outputs,
                                     void *workspace, cudaStream_t stream);
extern "C" int convertCPU_GPU(void *pCPU, void *pGPU, int size, int type,
                              int direction, cudaStream_t stream,
                              void *pBuffer);

#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

inline unsigned int elementSize(DataType t) {
  switch (t) {
    case DataType::kINT32:
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
  }
  return 0;
}

inline int getC(const Dims &d) { return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1; }

inline int getH(const Dims &d) { return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1; }

inline int getW(const Dims &d) { return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1; }

template <DataType in, DataType out>
void transform(const void *src, void *dst, int count) {
  assert(in == out);
  memcpy(dst, src, count * elementSize(in));
}

template <>
void transform<DataType::kINT8, DataType::kFLOAT>(const void *src, void *dst,
                                                  int count) {
  auto srcPtr = static_cast<const int8_t *>(src);
  auto dstPtr = static_cast<float *>(dst);
  std::transform(srcPtr, srcPtr + count, dstPtr,
                 [](int8_t in) { return static_cast<float>(in); });
}

template <>
void transform<DataType::kFLOAT, DataType::kINT8>(const void *src, void *dst,
                                                  int count) {
  auto srcPtr = static_cast<const float *>(src);
  auto dstPtr = static_cast<int8_t *>(dst);
  std::transform(srcPtr, srcPtr + count, dstPtr, [](float x) {
    x = std::max(x, float(INT8_MIN));
    x = std::min(x, float(INT8_MAX));
    return static_cast<int8_t>(x);
  });
}

#if NV_TENSORRT_MAJOR >= 6
class UpsamplePluginV2 : public IPluginV2IOExt {
 public:
  UpsamplePluginV2(const PluginFieldCollection &fc) { (void)fc; }

  UpsamplePluginV2(const void *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; ++i) {
      mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
      mOutputDims.d[i] = read<int>(d);
    }
    mDataType = static_cast<DataType>(read<int>(d));
    if (mDataType == DataType::kINT8) {
      mInHostScale = read<float>(d);
      mOutHostScale = read<float>(d);
    }
    mHScale = (float)mOutputDims.d[1] / mInputDims.d[1];
    mWScale = (float)mOutputDims.d[2] / mInputDims.d[2];
    if (d != a + length) DPRINTF(1, "UpsamplePluginV2 init error!\n");
    DPRINTF(1, "Scale=%.2f Type=%d \n", mHScale, (int)mDataType);
  }

  UpsamplePluginV2(float wscale, float hscale) {
    mWScale = wscale;
    mHScale = hscale;
  }

  UpsamplePluginV2() {}

  virtual ~UpsamplePluginV2() {}

 public:
  int getNbOutputs() const override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    int height = inputs[0].d[1] * mHScale;
    int width = inputs[0].d[2] * mWScale;
    return Dims3(inputs[0].d[0], height, width);
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return UpsampleV2Forward(batchSize, mHScale, mInputDims, mOutputDims,
                             mDataType, inputs, outputs, workspace, stream);
  }

  size_t getSerializationSize() const override {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(static_cast<int>(mDataType));
    if (mDataType == DataType::kINT8) {
      serializationSize += sizeof(float) * 2;
    }
    return serializationSize;
  }

  void serialize(void *buffer) const override {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, mInputDims.nbDims);
    assert(mInputDims.nbDims <= mInputDims.MAX_DIMS);
    for (int i = 0; i < mInputDims.nbDims; ++i) {
      write(d, mInputDims.d[i]);
    }
    write(d, mOutputDims.nbDims);
    assert(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
      write(d, mOutputDims.d[i]);
    }
    write(d, static_cast<int>(mDataType));
    if (mDataType == DataType::kINT8) {
      write(d, mInHostScale);
      write(d, mOutHostScale);
    }
    if (d != a + getSerializationSize())
      DPRINTF(1, "UpsamplePluginV2 serialize error!\n");
  }

  using IPluginV2IOExt::configurePlugin;
  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) override {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].type == out[0].type);
    assert(in[0].format == TensorFormat::kLINEAR &&
           out[0].format == TensorFormat::kLINEAR);

    mDataType = in[0].type;
    mInputDims = in[0].dims;
    mOutputDims = out[0].dims;
    mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
    mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
  }

  //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) const override {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= inOut[pos].type != DataType::kINT32;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes,
                             int nbInputs) const override {
    assert(inputTypes && nbInputs == 1);
    (void)index;
    return inputTypes[0];
  }

  const char *getPluginType() const override { return "UpsampleV2"; }

  const char *getPluginVersion() const override { return "2"; }

  void destroy() override { delete this; }

  IPluginV2Ext *clone() const override {
    auto *plugin = new UpsamplePluginV2(*this);
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.data(); }

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool *inputIsBroadcasted,
                                    int nbInputs) const override {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int inputIndex) const override {
    return false;
  }

 private:
  template <typename T>
  void write(char *&buffer, const T &val) const {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T>
  T read(const char *&buffer) const {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
  }

  void copyDeviceInputToFP32(const void *src, void *&dst) {
    assert(mDataType == DataType::kINT8);
    size_t inCount = getC(mInputDims) * getH(mInputDims) * getW(mInputDims);
    std::unique_ptr<char> inputTmp{new char[inCount * elementSize(mDataType)]};
    CHECK(cudaMemcpy(inputTmp.get(), src, inCount * elementSize(mDataType),
                     cudaMemcpyDeviceToHost));
    std::unique_ptr<float> inputFP32{new float[inCount]};
    transform<DataType::kINT8, DataType::kFLOAT>(inputTmp.get(),
                                                 inputFP32.get(), inCount);
    // int8 scale
    int hw = mInputDims.d[1] * mInputDims.d[2];
    for (int j = 0; j < mInputDims.d[0]; ++j) {
      std::transform(inputFP32.get() + hw * j, inputFP32.get() + hw * (j + 1),
                     inputFP32.get() + hw * j,
                     [&](float in) -> float { return in * mInHostScale; });
    }
    CHECK(cudaMalloc(&dst, inCount * elementSize(DataType::kFLOAT)));
    CHECK(cudaMemcpy(dst, inputFP32.get(),
                     inCount * elementSize(DataType::kFLOAT),
                     cudaMemcpyHostToDevice));
  }

  void copyDeviceToInt8Output(const void *src, void *dst) {
    size_t outCount = getC(mOutputDims) * getH(mOutputDims) * getW(mOutputDims);
    std::unique_ptr<float> outTmp{new float[outCount]};
    CHECK(cudaMemcpy(outTmp.get(), src,
                     outCount * elementSize(DataType::kFLOAT),
                     cudaMemcpyDeviceToHost));
    std::unique_ptr<char> outInt8{
        new char[outCount * elementSize(DataType::kINT8)]};
    // int8 + scale
    int hw = mOutputDims.d[1] * mOutputDims.d[2];
    for (int j = 0; j < mInputDims.d[0]; ++j) {
      std::transform(outTmp.get() + hw * j, outTmp.get() + hw * (j + 1),
                     outTmp.get() + hw * j,
                     [&](float in) -> float { return in / mOutHostScale; });
    }
    transform<DataType::kFLOAT, DataType::kINT8>(outTmp.get(), outInt8.get(),
                                                 outCount);
    CHECK(cudaMemcpy(dst, outInt8.get(), outCount, cudaMemcpyHostToDevice));
  }

 private:
  DataType mDataType;
  float mHScale{2.0f}, mWScale{2.0f};
  Dims mInputDims;
  Dims mOutputDims;
  float mInHostScale{-1.0f};
  float mOutHostScale{-1.0f};
  std::string mNamespace;
};

// Batch Pad Concatenate op for main/narrow combined onnx
class BatchPadConcatPlugin : public IPluginV2IOExt {
 public:
  BatchPadConcatPlugin(const PluginFieldCollection &fc) { (void)fc; }

  BatchPadConcatPlugin(const void *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; ++i) {
      mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
      mOutputDims.d[i] = read<int>(d);
    }

    mInDataType = static_cast<DataType>(read<int>(d));
    mOutDataType = static_cast<DataType>(read<int>(d));
    mBatch = read<int>(d);
    mWsize = read<int>(d);
    CPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++) {
      CPUWeights[nb].resize(mWsize);
      float *pW = &CPUWeights[nb][0];
      read<float>(d, pW, mWsize);
    }

    if (mInDataType == DataType::kINT8) {
      mInHostScale = read<float>(d);
      mOutHostScale = read<float>(d);
    }
    mHPad = mOutputDims.d[1] - mInputDims.d[1];
    mWPad = mOutputDims.d[2] - mInputDims.d[2];
    if (d != a + length) DPRINTF(1, "BatchPadConcatPlugin init error!\n");
    DPRINTF(1, "%s InType=%d OutType=%d \n", __func__, (int)mInDataType,
            (int)mOutDataType);
  }

  int mBatch = 0;
  int mWsize = 0;
  std::vector<std::vector<float>> CPUWeights;
  void *GPUWeights[16];
  BatchPadConcatPlugin(int batch, std::vector<Weights> &weights, Dims shape) {
    mBatch = batch;
    mHPad = 0;
    mWPad = 0;

    // copy weight from onnx file to CPU
    int wnum = weights.size() / mBatch;
    int wsize = weights[0].count;
    mCPad = wnum * shape.d[0];
    mWsize = wsize * wnum;
    CPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++) {
      CPUWeights[nb].resize(mWsize);
      float *pW = &CPUWeights[nb][0];
      for (int i = 0; i < wnum; i++) {
        memcpy(pW, weights[nb * wnum + i].values, wsize * sizeof(float));
        pW += wsize;
      }
    }
    DPRINTF(1, "%s mBatch=%d mCPad=%d\n", __func__, mBatch, mCPad);
  }

  BatchPadConcatPlugin() {}

  virtual ~BatchPadConcatPlugin() {}

 public:
  int getNbOutputs() const override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) override {
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    int ch = inputs[0].d[0] + mCPad;
    int height = inputs[0].d[1] + mHPad;
    int width = inputs[0].d[2] + mWPad;
    return Dims3(ch, height, width);
  }

  int initialize() override {
    DPRINTF(2, "%s %d\n", __FUNCTION__, __LINE__);
    // copy weights from CPU to GPU buffer
    void *pBuffer;
    CHECK(cudaMalloc(&pBuffer, mWsize * sizeof(float)));
    for (int nb = 0; nb < mBatch; nb++) {
      float *pW = nullptr;
      if (mOutDataType == DataType::kHALF) {
        CHECK(cudaMalloc(&pW, mWsize * sizeof(__half)));
      } else {
        CHECK(cudaMalloc(&pW, mWsize * sizeof(float)));
      }
      convertCPU_GPU(&CPUWeights[nb][0], pW, mWsize, (int)mOutDataType, 0,
                     (cudaStream_t)0x2, pBuffer);
      GPUWeights[nb] = pW;
    }
    cudaFree(pBuffer);
    DPRINTF(2, "%s %d\n", __FUNCTION__, __LINE__);
    return 0;
  }

  void terminate() override {
    for (int nb = 0; nb < mBatch; nb++) {
      cudaFree(GPUWeights[nb]);
      GPUWeights[nb] = nullptr;
    }
  }

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return BatchPadConcatForward(batchSize, mInputDims, mOutputDims,
                                 mInDataType, mOutDataType, GPUWeights, inputs,
                                 outputs, workspace, stream);
  }

  size_t getSerializationSize() const override {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(static_cast<int>(mInDataType));
    serializationSize += sizeof(static_cast<int>(mOutDataType));
    serializationSize += sizeof(mBatch);
    serializationSize += sizeof(mWsize);
    serializationSize += mBatch * mWsize * sizeof(float);

    if (mInDataType == DataType::kINT8) {
      serializationSize += sizeof(float) * 2;
    }
    return serializationSize;
  }

  void serialize(void *buffer) const override {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, mInputDims.nbDims);
    assert(mInputDims.nbDims <= mInputDims.MAX_DIMS);
    for (int i = 0; i < mInputDims.nbDims; ++i) {
      write(d, mInputDims.d[i]);
    }
    write(d, mOutputDims.nbDims);
    assert(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
    for (int i = 0; i < mOutputDims.nbDims; ++i) {
      write(d, mOutputDims.d[i]);
    }

    write(d, static_cast<int>(mInDataType));
    write(d, static_cast<int>(mOutDataType));
    write(d, static_cast<int>(mBatch));
    write(d, static_cast<int>(mWsize));
    for (int nb = 0; nb < mBatch; nb++) {
      write(d, &CPUWeights[nb][0], mWsize);
    }

    if (mInDataType == DataType::kINT8) {
      write(d, mInHostScale);
      write(d, mOutHostScale);
    }
    if (d != a + getSerializationSize())
      DPRINTF(1, "BatchPadConcatPlugin serialize error!\n");
  }

  using IPluginV2IOExt::configurePlugin;
  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) override {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].format == TensorFormat::kLINEAR &&
           out[0].format == TensorFormat::kLINEAR);

    mInDataType = in[0].type;
    mOutDataType = out[0].type;
    mInputDims = in[0].dims;
    mOutputDims = out[0].dims;
    mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
    mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
    DPRINTF(1, "configType: In=%d Out=%d \n", (int)mInDataType,
            (int)mOutDataType);
  }

  //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
  // IOType: FLOAT, HALF, INT8->HALF,
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) const override {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= inOut[pos].type != DataType::kINT32;
    if (pos == 1) {
      condition &= inOut[1].type != DataType::kINT8;
      condition &= (inOut[1].type == inOut[0].type) ||
                   (DataType::kINT8 == inOut[0].type &&
                    DataType::kHALF == inOut[1].type);
    }
    return condition;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes,
                             int nbInputs) const override {
    assert(inputTypes && nbInputs == 1);
    if (DataType::kINT8 == inputTypes[0])
      return DataType::kHALF;
    else
      return inputTypes[0];
  }

  const char *getPluginType() const override { return "BatchConcatPad"; }

  const char *getPluginVersion() const override { return "1"; }

  void destroy() override { delete this; }

  IPluginV2Ext *clone() const override {
    auto *plugin = new BatchPadConcatPlugin(*this);
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) override {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.data(); }

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool *inputIsBroadcasted,
                                    int nbInputs) const override {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int inputIndex) const override {
    return false;
  }

 private:
  template <typename T>
  void write(char *&buffer, const T &val) const {
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T>
  void write(char *&buffer, T *data, int count) const {
    memcpy(buffer, data, sizeof(T) * count);
    buffer += sizeof(T) * count;
  }

  template <typename T>
  T read(const char *&buffer) const {
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
  }

  template <typename T>
  void read(const char *&buffer, T *data, int count) const {
    memcpy(data, buffer, sizeof(T) * count);
    buffer += sizeof(T) * count;
  }

 private:
  DataType mInDataType;
  DataType mOutDataType;
  Dims mInputDims;
  Dims mOutputDims;
  int mCPad, mHPad, mWPad;
  float mInHostScale{-1.0f};
  float mOutHostScale{-1.0f};
  std::string mNamespace;
};

#endif