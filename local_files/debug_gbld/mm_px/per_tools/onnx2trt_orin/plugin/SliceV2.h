/*
 * Copyright (c) 2021, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#ifndef __SliceV2_HPP__
#define __SliceV2_HPP__

#include "common.h"
using namespace std;

extern "C" int SliceV2Forward(int batchSize, Dims &inDims, Dims outDims, DataType inType, DataType outType,
                            TensorFormat inFormat, TensorFormat outFormat, int *starts, int *ends,
                            const void *const *inputs, void *const*outputs, void *workspace, cudaStream_t stream);

// Takes idx from [MIN_INT, MAX_INT] to [0, ax_size] (for Slice op)
inline int slice_clip_index(int ax_size, int idx) {
  if (idx < 0) {
    idx += (ax_size + 1);
  }
  return std::min(std::max(idx, 0), ax_size);
}

//Slice plugin for safety PDK526
class SliceV2Plugin : public IPluginV2IOExt{
public:
  SliceV2Plugin(const PluginFieldCollection *fc) {
    const PluginField* fields = fc->fields;
    std::vector<int> axes{0};    
    std::vector<int> starts{0};
    std::vector<int> ends{0};
    for (auto i = 0; i < fc->nbFields; ++i)
    {  
      const char* attrName = fields[i].name;
      DPRINTF(2, "SliceV2Plugin attrName=%s\n", attrName);
      if (!strcmp(attrName, "axes")) {  
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          axes.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              axes[j] = *aR;
              DPRINTF(3, "axes[%d]=%d\n", j, axes[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "starts")) {  
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          starts.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              starts[j] = *aR;
              DPRINTF(3, "starts[%d]=%d\n", j, starts[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "ends")) {  
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          ends.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              ends[j] = *aR;
              DPRINTF(3, "ends[%d]=%d\n", j, ends[j]);
              aR++;
          }
        }
      } 
    }
    
    for ( int i = 0; i< 3; i++){
      mStart_CHW[i] = 0;
      mEnd_CHW[i] = -1;
    }
    for( int i = 0; i < axes.size(); i++){
      assert(axes[i] >= 1 && axes[i] <= 3); // CHW, No Batch channel
      mStart_CHW[axes[i]-1] = starts[i];
      mEnd_CHW[axes[i]-1] = ends[i];       
    }
  }

  SliceV2Plugin(const void *data, size_t length)
  {
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
    for ( int i = 0; i< 3; i++){
      mStart_CHW[i] = read<int>(d);
      mEnd_CHW[i] = read<int>(d);
    }   
    mInDataType = static_cast<DataType>(read<int>(d));
    mOutDataType = static_cast<DataType>(read<int>(d));
    mInTensorFormat = static_cast<TensorFormat>(read<int>(d));
    mOutTensorFormat = static_cast<TensorFormat>(read<int>(d));
    
    if (d != a + length)
      DPRINTF(1, "SliceV2Plugin init error!\n");
  }

  SliceV2Plugin() {}

  virtual ~SliceV2Plugin() {}

public:
  int getNbOutputs() const noexcept override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override
  {
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    Dims3 outSize;
    for(int i=0; i< 3; i++){
      mStart_CHW[i] = slice_clip_index(inputs[0].d[i], mStart_CHW[i]);
      mEnd_CHW[i]   = slice_clip_index(inputs[0].d[i], mEnd_CHW[i]);
      outSize.d[i] = mEnd_CHW[i] - mStart_CHW[i];
      DPRINTF(2, "SliceV2 inter params = %d %d %d\n", mStart_CHW[i], mEnd_CHW[i], outSize.d[i]);
    }      
    DPRINTF(2, "SliceV2 getOutputDimensions = %d %d %d\n", outSize.d[0], outSize.d[1], outSize.d[2]);
    return outSize;
  }

  int initialize() noexcept override { return 0; }
  void terminate() noexcept override { return; }

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void *const*outputs, void *workspace, cudaStream_t stream) noexcept override
  {
    if (TRT_DEBUGLEVEL == -1) {
            printf("Skip SliceV2::enqueue!!\n");
            return 0;
    } 
    
    return SliceV2Forward(batchSize, mInputDims, mOutputDims, mInDataType, mOutDataType, mInTensorFormat,
                                   mOutTensorFormat, mStart_CHW, mEnd_CHW, inputs, outputs, workspace, stream);
  }

  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(mStart_CHW) + sizeof(mEnd_CHW);
    serializationSize += sizeof(static_cast<int>(mInDataType));
    serializationSize += sizeof(static_cast<int>(mOutDataType));
    serializationSize += sizeof(static_cast<int>(mInTensorFormat));
    serializationSize += sizeof(static_cast<int>(mOutTensorFormat));

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
    for ( int i = 0; i< 3; i++){
      write(d, mStart_CHW[i]);
      write(d, mEnd_CHW[i]);
    }
    write(d, static_cast<int>(mInDataType));
    write(d, static_cast<int>(mOutDataType));
    write(d, static_cast<int>(mInTensorFormat));
    write(d, static_cast<int>(mOutTensorFormat));
    if (d != a + getSerializationSize())
      DPRINTF(1, "SliceV2Plugin serialize error!\n");
  }

  void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].format == TensorFormat::kLINEAR);
    assert(out[0].format == TensorFormat::kLINEAR || out[0].format == TensorFormat::kCHW4 ||
           out[0].format == TensorFormat::kCHW32);

    mInDataType = in[0].type;
    mOutDataType = out[0].type;
    mInTensorFormat = in[0].format;
    mOutTensorFormat = out[0].format;
    mInputDims = in[0].dims;
    mOutputDims = out[0].dims;

    DPRINTF(2, "SliceV2: %s Datatype: In=%d Out=%d, Format In=%d Out=%d \n", __func__, (int)mInDataType, (int)mOutDataType,
            (int)mInTensorFormat, (int)mOutTensorFormat);
    DPRINTF(2, "SliceV2: Input dim: 1=%d, 2=%d, 3=%d; Output dim: 1=%d, 2=%d, 3=%d\n", in[0].dims.d[0], in[0].dims.d[1],
            in[0].dims.d[2], out[0].dims.d[0], out[0].dims.d[1], out[0].dims.d[2]);
  }

  //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
  // IOType: FLOAT, HALF, INT8->HALF,
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = (inOut[0].format == TensorFormat::kLINEAR);
    if (1 == pos)
    {
      //condition &= ((inOut[1].format == TensorFormat::kLINEAR) ||
      //              ((inOut[1].format == TensorFormat::kCHW4) && (inOut[1].type == DataType::kINT8)) ||
      //              ((inOut[1].format == TensorFormat::kCHW32) && (inOut[1].type == DataType::kINT8)));
      condition &= (inOut[1].format == TensorFormat::kLINEAR);      
      condition &= (inOut[1].type == inOut[0].type);
    }
    DPRINTF(2, "SliceV2: In type: %d, format: %d; Out type: %d, format: %d  Condition: %d\n", (int)inOut[0].type, (int)inOut[0].format,
            (int)inOut[1].type, (int)inOut[1].format, (int)condition);
    DPRINTF(2, "SliceV2: Input dim: 1=%d, 2=%d, 3=%d; Output dim: 1=%d, 2=%d, 3=%d\n", inOut[0].dims.d[1], inOut[0].dims.d[2],
            inOut[0].dims.d[3], inOut[1].dims.d[1], inOut[1].dims.d[2], inOut[1].dims.d[3]);
    return condition;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
  {
    assert(inputTypes && nbInputs == 1);
    return inputTypes[0];
    // if( DataType::kINT8 == inputTypes[0] ) return DataType::kHALF;
    // else return inputTypes[0];
  }

  const char *getPluginType() const noexcept override { return "SliceV2"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override{
    auto *plugin = new SliceV2Plugin(*this);
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
  DataType mInDataType;
  DataType mOutDataType;
  TensorFormat mInTensorFormat;
  TensorFormat mOutTensorFormat;
  Dims mInputDims;
  Dims mOutputDims;
  int mStart_CHW[3];
  int mEnd_CHW[3];

  std::string mNamespace;
};
class SliceV2PluginCreator : public IPluginCreator
{
public:
  SliceV2PluginCreator(){
    mPluginAttributes.emplace_back(PluginField("axes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ends", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("starts", nullptr, PluginFieldType::kINT32, 1));
      
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }
  
  const char *getPluginName() const noexcept override { return "SliceV2"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    auto plugin = new SliceV2Plugin(fc);
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new SliceV2Plugin(serialData, serialLength);
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


REGISTER_TENSORRT_PLUGIN(SliceV2PluginCreator);

#endif // __SliceV2_HPP__