#ifndef __RESIZENEAREST_HPP__
#define __RESIZENEAREST_HPP__

#include "common.h"
// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const &dims)
{
  //printf("Resize %d, [%d, %d, %d]\n", dims.nbDims, dims.d[0], dims.d[1], dims.d[2]);fflush(stdout);
  //printf("Resize %d, [%d, %d, %d]\n", dims.nbDims, dims.type[0], dims.type[1], dims.type[2]);fflush(stdout);
  return (dims.nbDims == 3);
}
extern "C" int ResizeNearestForward(int batchSize, const void *const *inputs, void *const*outputs, void *workspace,
                                    cudaStream_t stream, Dims input_dims, Dims _output_dims, float _scale[], int _ndims, bool type_float);
class ResizeNearestPluginV2 : public IPluginV2IOExt
{

public:
  ResizeNearestPluginV2(const PluginFieldCollection &fc) { (void)fc; }
  ResizeNearestPluginV2(std::vector<float> const &scale)
      : _ndims(scale.size())
  {
    assert(scale.size() <= nvinfer1::Dims::MAX_DIMS);
    std::copy(scale.begin(), scale.end(), _scale);
  }
  ResizeNearestPluginV2(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    _ndims = read<int>(d);
    for (int i = 0; i < Dims::MAX_DIMS; ++i)
    {
      _scale[i] = read<float>(d);
    }
    _input_dims = read<Dims>(d);
    _output_dims = read<Dims>(d);
    _data_type = read<int>(d);
    if (d != a + length)
      DPRINTF(1, "ResizeNearestPluginV2 init error!\n");
    DPRINTF(1, "ResizeNearestPluginV2 _ndims=%d _scale[0]=%f \n", _ndims, _scale[0]);
  }
  ~ResizeNearestPluginV2() {}

private:
  int _ndims;
  float _scale[Dims::MAX_DIMS];
  Dims _output_dims;
  Dims _input_dims;
  int _data_type;
  std::string mNamespace;

public:
  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(int);                  //_ndims
    serializationSize += sizeof(int) * Dims::MAX_DIMS; //_scale
    serializationSize += sizeof(Dims);                 //_input_dims
    serializationSize += sizeof(Dims);                 //_output_dims
    serializationSize += sizeof(int);                  //_data_type
    return serializationSize;
  }

  void serialize(void *buffer) const noexcept override
  {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, _ndims);
    for (int i = 0; i < Dims::MAX_DIMS; ++i)
    {
      write(d, _scale[i]);
    }
    write(d, _input_dims);
    write(d, _output_dims);
    write(d, _data_type);
    if (d != a + getSerializationSize())
      DPRINTF(1, "ResizeNearestPluginV2 serialize error!\n");
  }

protected:
  int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace,
              cudaStream_t stream) noexcept override
  {
    if (TRT_DEBUGLEVEL == -1)
    {
      printf("Skip ResizeNearestPluginV2::enqueue!!\n");
      return 0;
    }
    bool type_float = (_data_type == (int)(DataType::kFLOAT));
    return ResizeNearestForward(batchSize, inputs, outputs, workspace, stream, _input_dims, _output_dims, _scale, _ndims, type_float);
  }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override
  {
    _input_dims = inputDims[0];
    assert(is_CHW(_input_dims));
    assert(_ndims == 2);
    assert(index == 0);
    nvinfer1::Dims output;
    output.nbDims = _input_dims.nbDims;
    int s = 0;
    for (int d = 0; d < _input_dims.nbDims; ++d)
    {
      //output.type[d] = _input_dims.type[d];
      if (d != 0)
      {
        output.d[d] = int(_input_dims.d[d] * _scale[s++]);
      }
      else
      {
        output.d[d] = _input_dims.d[d];
      }
    }
    _output_dims = output;
    return output;
  }

  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}

  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].type == out[0].type);
    _data_type = (int)out[0].type;
    assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
  }
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= inOut[pos].type == inOut[0].type;
    condition &= (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF);
    return condition;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
  {
    assert(inputTypes && nbInputs == 1);
    (void)index;
    return inputTypes[0];
  }
  size_t getWorkspaceSize(int maxBatchSize) const noexcept override
  {
    int nsize = 1024 * sizeof(bool);
    DPRINTF(2, "ResizeNearest getWorkspaceSize = %d\n", nsize);
    return nsize;
  }
  const char *getPluginType() const noexcept override { return "ResizeNearest"; }
  const char *getPluginVersion() const noexcept override { return "1"; }
  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new ResizeNearestPluginV2(*this);
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.data(); }

  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override
  {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }
};
class ResizeNearestPluginV2Creator : public IPluginCreator
{
public:
  const char *getPluginName() const noexcept override { return "ResizeNearest"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFieldCollection; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    auto plugin = new ResizeNearestPluginV2(*fc);
    mFieldCollection = *fc;
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new ResizeNearestPluginV2(serialData, serialLength);
    mPluginName = name;
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
  std::string mNamespace;
  std::string mPluginName;
  PluginFieldCollection mFieldCollection{0, nullptr};
};

REGISTER_TENSORRT_PLUGIN(ResizeNearestPluginV2Creator);
#endif //__RESIZENEAREST_HPP__