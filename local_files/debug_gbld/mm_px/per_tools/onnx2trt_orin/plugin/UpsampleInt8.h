/*
 * Copyright (c) 2021, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#ifndef __UPSAMPLEINT8_HPP__
#define __UPSAMPLEINT8_HPP__

#include "common.h"
using namespace std;

extern "C" int UpsampleV2ForwardV2(int batchSize, float mWScale, float mHSale, Dims &mInputDims, Dims &mOutputDims, DataType &mDataType,
                                   TensorFormat inTensorFormat, TensorFormat outTensorFormat, const void *const *inputs,
                                   void *const*outputs, void *workspace, cudaStream_t stream);

inline int getC(const Dims &d) { return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1; }

inline int getH(const Dims &d) { return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1; }

inline int getW(const Dims &d) { return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1; }

template <DataType in, DataType out>
void transform(const void *src, void *dst, int count);
// {
//   assert(in == out);
//   memcpy(dst, src, count * elementSize(in));
// }

template <>
void transform<DataType::kINT8, DataType::kFLOAT>(const void *src, void *dst, int count);
// {
//   auto srcPtr = static_cast<const int8_t *>(src);
//   auto dstPtr = static_cast<float *>(dst);
//   std::transform(srcPtr, srcPtr + count, dstPtr, [](int8_t in) { return static_cast<float>(in); });
// }

template <>
void transform<DataType::kFLOAT, DataType::kINT8>(const void *src, void *dst, int count);
// {
//   auto srcPtr = static_cast<const float *>(src);
//   auto dstPtr = static_cast<int8_t *>(dst);
//   std::transform(srcPtr, srcPtr + count, dstPtr, [](float x) {
//     x = std::max(x, float(INT8_MIN));
//     x = std::min(x, float(INT8_MAX));
//     return static_cast<int8_t>(x);
//   });
// }

class UpsamplePluginV2 : public IPluginV2IOExt
{
public:
  UpsamplePluginV2(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; ++i)
    {
      mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; ++i)
    {
      mOutputDims.d[i] = read<int>(d);
    }
    mDataType = static_cast<DataType>(read<int>(d));
    mInTensorFormat = static_cast<TensorFormat>(read<int>(d));
    mOutTensorFormat = static_cast<TensorFormat>(read<int>(d));
    if (mDataType == DataType::kINT8)
    {
      mInHostScale = read<float>(d);
      mOutHostScale = read<float>(d);
    }
    mHScale = (float)mOutputDims.d[1] / mInputDims.d[1];
    mWScale = (float)mOutputDims.d[2] / mInputDims.d[2];
    if (d != a + length)
      DPRINTF(1, "UpsamplePluginV2 init error!\n");
    DPRINTF(1, "UpsamplePluginV2 mWScale=%.3f mHScale=%.3f Type=%d \n", mWScale, mHScale, (int)mDataType);
  }

  UpsamplePluginV2(float wscale, float hscale)
  {
    mWScale = wscale;
    mHScale = hscale;
  }

  UpsamplePluginV2() {}

  virtual ~UpsamplePluginV2() {}

public:
  int getNbOutputs() const noexcept override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override
  {
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    int height = inputs[0].d[1] * mHScale;
    int width = inputs[0].d[2] * mWScale;
    return Dims3(inputs[0].d[0], height, width);
  }

  int initialize() noexcept override { return 0; }

  void terminate() noexcept override {}

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override
  {
    return UpsampleV2ForwardV2(batchSize, mWScale, mHScale, mInputDims, mOutputDims, mDataType, mInTensorFormat,
                               mOutTensorFormat, inputs, outputs, workspace, stream);
  }

  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(static_cast<int>(mDataType));
    serializationSize += sizeof(static_cast<int>(mInTensorFormat));
    serializationSize += sizeof(static_cast<int>(mOutTensorFormat));
    if (mDataType == DataType::kINT8)
    {
      serializationSize += sizeof(float) * 2;
    }
    return serializationSize;
  }

  void serialize(void *buffer) const noexcept override
  {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, mInputDims.nbDims);
    assert(mInputDims.nbDims <= mInputDims.MAX_DIMS);
    for (int i = 0; i < mInputDims.nbDims; ++i)
    {
      write(d, mInputDims.d[i]);
    }
    write(d, mOutputDims.nbDims);
    assert(mOutputDims.nbDims <= mOutputDims.MAX_DIMS);
    for (int i = 0; i < mOutputDims.nbDims; ++i)
    {
      write(d, mOutputDims.d[i]);
    }
    write(d, static_cast<int>(mDataType));
    write(d, static_cast<int>(mInTensorFormat));
    write(d, static_cast<int>(mOutTensorFormat));
    if (mDataType == DataType::kINT8)
    {
      write(d, mInHostScale);
      write(d, mOutHostScale);
    }
    if (d != a + getSerializationSize())
      DPRINTF(1, "UpsamplePluginV2 serialize error!\n");
  }

  void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].type == out[0].type);
    assert(in[0].type == DataType::kFLOAT || in[0].type == DataType::kHALF || in[0].type == DataType::kINT8);

    assert(in[0].format == out[0].format);
    assert(in[0].format == TensorFormat::kLINEAR || in[0].format == TensorFormat::kCHW32);

    mDataType = in[0].type;
    mInTensorFormat = in[0].format;
    mInputDims = in[0].dims;
    mOutTensorFormat = out[0].format;
    mOutputDims = out[0].dims;
    mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
    mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
  }

  //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool conditionCommon = inOut[pos].format == TensorFormat::kLINEAR;
    conditionCommon &= (inOut[pos].type  == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
    conditionCommon &= inOut[pos].type   == inOut[0].type;
    conditionCommon &= inOut[pos].format == inOut[0].format;

    bool conditionCHW32 = inOut[pos].format == TensorFormat::kCHW32;
    conditionCHW32 &= inOut[pos].type   == DataType::kINT8;
    conditionCHW32 &= inOut[pos].type   == inOut[0].type;
    conditionCHW32 &= inOut[pos].format == inOut[0].format;

    return conditionCommon || conditionCHW32;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
  {
    assert(inputTypes && nbInputs == 1);
    (void)index;
    return inputTypes[0];
  }

  const char *getPluginType() const noexcept override { return "UpsampleV2"; }

  const char *getPluginVersion() const noexcept override { return "2"; }

  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new UpsamplePluginV2(*this);
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.data(); }

  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override
  {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }

private:
  DataType mDataType;
  TensorFormat mInTensorFormat;
  TensorFormat mOutTensorFormat;
  float mHScale{2.0f}, mWScale{2.0f};
  Dims mInputDims;
  Dims mOutputDims;
  float mInHostScale{-1.0f};
  float mOutHostScale{-1.0f};
  std::string mNamespace;
};

class UpsamplePluginV2Creator : public IPluginCreator
{
public:
  UpsamplePluginV2Creator(){
    mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 4));
            
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }
  
  const char *getPluginName() const noexcept override { return "UpsampleV2"; }

  const char *getPluginVersion() const noexcept override { return "2"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    const PluginField* fields = fc->fields;
    float scales[4] = {1.f, 1.f, 1.f, 1.f};
    std::vector<float> cam_weight[2][3];
    for (auto i = 0; i < fc->nbFields; ++i)
    {  
      const char* attrName = fields[i].name;
      DPRINTF(2, "UpsamplePluginV2Creator attrName=%s length=%d\n", attrName, fields[i].length);
      if (!strcmp(attrName, "scales")) {  
        assert(fields[i].type == PluginFieldType::kFLOAT32);
        int32_t size = fields[i].length/sizeof(float);
        if(size < 4)
          size = fields[i].length;
        if (size > 0) {
          const auto* aR = static_cast<const float*>(fields[i].data);
          for (auto j = 0; j < size && j < 4; j++) {
              scales[j] = *aR;
              DPRINTF(2, "scales[%d]=%f\n", j, scales[j]);
              aR++;
          }
        }
      } 
    }
      
    auto plugin = new UpsamplePluginV2(scales[3], scales[2]);
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new UpsamplePluginV2(serialData, serialLength);
    mPluginName = name;
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
  std::string mNamespace;
  std::string mPluginName;
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes; 
};

REGISTER_TENSORRT_PLUGIN(UpsamplePluginV2Creator);
#endif // __UPSAMPLEINT8_HPP__
