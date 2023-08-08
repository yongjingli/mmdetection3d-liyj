
#ifndef __CONV_LSTM_HPP__
#define __CONV_LSTM_HPP__

#include <unordered_map>
#include "common.h"

using namespace nvinfer1;

extern "C" int SetConvLSTMState(cudaStream_t stream, int state);
extern "C" int setConvLSTMBuff(void* ptr, cudaStream_t stream, int magic);
extern "C" int ConvLSTMPluginForward(int batchSize, const void *const *inputs, void *const*outputs, void *workspace,
                                     cudaStream_t stream, int params_int[2], void **params_pointer[2], Dims &input_dims, std::vector<int> &_attrs,
                                     DataType mDataType, TensorFormat mTensorFormat, float mInputScale, float mOutputScale);
class ConvLSTMPluginV2 : public IPluginV2IOExt
{
public:
  ConvLSTMPluginV2(const PluginFieldCollection &fc) { (void)fc; }

  ConvLSTMPluginV2(int mode, std::vector<int> &attrs)
  {
    _mode = mode;
    _attrs = attrs;
    _attrs_num = attrs.size();
    DPRINTF(1, "ConvLSTMPluginV2(mode:%d,attributs:%zu,magic:%d)\n", _mode, _attrs.size(),_attrs[3]);
  }
  ConvLSTMPluginV2(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    _mode = read<int>(d);
    _attrs_num = read<int>(d);
    _attrs.resize(_attrs_num);
    for (int i = 0; i < _attrs_num; ++i)
    {
      _attrs[i] = read<int>(d);
    }
    _input_dims = read<Dims>(d);

    if (d + sizeof(int) * 4 <= a + length) {
      _mDataType     = static_cast<DataType>(read<int>(d));
      _mTensorFormat = static_cast<TensorFormat>(read<int>(d));
      _mInputScale   = read<float>(d);
      _mOutputScale  = read<float>(d);
    }

    //set _magic
    if (_mode != 0)
    {
      _magic = _attrs[3];
    }
    if (d != a + length) {
      DPRINTF(1, "ConvLSTMPluginV2 init error!, The length is %zu\n", length);
      DPRINTF(1, "Pointer d is %p, pointer a is %p.\n", d, a);
    }

    DPRINTF(2, "ConvLSTMPluginV2 _mode=%d _attrs_num=%d _magic=%d\n", _mode, _attrs_num, _magic);
  }
  ~ConvLSTMPluginV2() {}

private:
  // layer arg, need serialize & deserialize
  // int _in_c, _hidden_c, _kernel, _repeat, _stride, _pad;
  int _mode = 0; // 0: all part  1: only concat, 2: only conv, 3: other
  int _attrs_num;
  std::vector<int> _attrs; // 6 int
  Dims _input_dims;
  // runtime device/GPU buffers
  bool _initialized = {false};
  void *d_Ct = nullptr;
  void *d_Ht = nullptr;
  int _magic = 0;
  std::string mNamespace;

  float        _mInputScale = 1.f;
  float        _mOutputScale= 1.f;
  DataType     _mDataType = DataType::kFLOAT; 
  TensorFormat _mTensorFormat = TensorFormat::kLINEAR; 

  cudaStream_t _stream = nullptr;

public:
  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(int);              //_mode
    serializationSize += sizeof(int);              //_attrs_num
    serializationSize += sizeof(int) * _attrs_num; //_attrs
    serializationSize += sizeof(Dims);             //_input_dims
    serializationSize += sizeof(int);              //_mDataType
    serializationSize += sizeof(int);              //_mTensorFormat
    serializationSize += sizeof(float);            //_mInputScale
    serializationSize += sizeof(float);            //_mOutputScale
    return serializationSize;
  }

  void serialize(void *buffer) const noexcept override
  {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, _mode);
    write(d, _attrs_num);
    for (int i = 0; i < _attrs_num; ++i)
    {
      write(d, _attrs[i]);
    }

    write(d, _input_dims);

    write(d, static_cast<int>(_mDataType));
    write(d, static_cast<int>(_mTensorFormat));
    write(d, _mInputScale);
    write(d, _mOutputScale);

    DPRINTF(1, "ConvLSTMPluginV2 Serialization Size is %zu, the attris num is %d.\n", getSerializationSize(), _attrs_num);

    if (d != a + getSerializationSize())
      DPRINTF(1, "ConvLSTMPluginV2 serialize error!\n");
  }

public:
  int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override
  {
    assert(_initialized);
    _stream = stream;
    int params_int[2] = {_mode, _magic};
    void **params_pointer[2] = {&d_Ct, &d_Ht};
    return ConvLSTMPluginForward(batchSize, inputs, outputs, workspace, stream,
                                 params_int, params_pointer, _input_dims, _attrs,
                                 _mDataType, _mTensorFormat, _mInputScale, _mOutputScale);
  }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override
  {
    assert(nbInputDims == 1);
    nvinfer1::Dims const &input = inputDims[0];
    _input_dims = inputDims[0];
    nvinfer1::Dims output = input;
    if (1 == _mode)
      output.d[0] = _attrs[0] + _attrs[1];
    if (2 == _mode || 3 == _mode)
      output.d[0] = _attrs[1];
    DPRINTF(1, "Input shape=%d(%d,%d,%d)\n", input.nbDims, input.d[0], input.d[1], input.d[2]);
    DPRINTF(1, "Output shape=%d(%d,%d,%d)\n", output.nbDims, output.d[0], output.d[1], output.d[2]);
    return output;
  }

  int initialize() noexcept override
  {
    if (_initialized)
    {
      return 0;
    }

    auto const &input_dims = _input_dims;
    int max_batch = 4;
    if (0 == _mode)
    {
      printf("Didn't support mode 0 (cudnn version)!!\n");
    }
    else
    {
      int hidden_ch = _attrs[0];
      int in_h = input_dims.d[1];
      int in_w = input_dims.d[2];
      int CtSize = max_batch * hidden_ch * in_h * in_w * sizeof(float);
      if (2 >= _mode) {
        CHECK_CUDA(cudaMalloc(&d_Ht, CtSize));
        CHECK_CUDA(cudaMemset(d_Ht, 0, CtSize));
      }
      else { // 3 == _mode
        CHECK_CUDA(cudaMalloc(&d_Ct, CtSize));
        CHECK_CUDA(cudaMemset(d_Ct, 0, CtSize));
      }
    }

    _initialized = true;

    return 0;
  }
  void terminate() noexcept override
  {
    if (!_initialized)
    {
      return;
    }

    if (2 >= _mode && nullptr != d_Ht)
    {
      setConvLSTMBuff(d_Ht, _stream, _magic);
    }
    if (3 == _mode && nullptr != d_Ct)
    {
      setConvLSTMBuff(d_Ct, _stream, _magic + 1);
    }

    _initialized = false;
  }
  size_t getWorkspaceSize(int maxBatchSize) const noexcept override
  {
    int nsize = 1024 * sizeof(float);
    DPRINTF(2, "ConvLSTMPlugin getWorkspaceSize = %d\n", nsize);
    return nsize;
  }

  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].type == out[0].type);
    assert(in[0].type == DataType::kFLOAT || in[0].type == DataType::kINT8);
    // assert(in[0].type == DataType::kFLOAT);
    assert(in[0].format == out[0].format);
    assert(in[0].format == TensorFormat::kLINEAR || in[0].format == TensorFormat::kCHW32);
    // assert(in[0].format == TensorFormat::kLINEAR);

    _mDataType    = in[0].type;
    _mTensorFormat= in[0].format;
    _mInputScale  = in[0] .scale;
    _mOutputScale = out[0].scale;

    DPRINTF(2, "ConvLSTM Mode %d, the input scale is %.5f, the output scale is %.5f.\n", _mInputScale, _mOutputScale);
    DPRINTF(2, "ConvLSTM Mode %d, the data type %d, the tensor format is %d.\n", static_cast<int>(_mDataType), static_cast<int>(_mTensorFormat));
  }

  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool conditionCommon = inOut[pos].format == TensorFormat::kLINEAR;
    conditionCommon &= inOut[pos].type   == DataType::kFLOAT;
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
  virtual const char *getPluginType() const noexcept override { return "ConvLSTM"; }
  const char *getPluginVersion() const noexcept override { return "2"; }

  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new ConvLSTMPluginV2(*this);
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
class ConvLSTMPluginV2Creator : public IPluginCreator
{
public:
  const char *getPluginName() const noexcept override { return "ConvLSTM"; }

  const char *getPluginVersion() const noexcept override { return "2"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFieldCollection; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    auto plugin = new ConvLSTMPluginV2(*fc);
    mFieldCollection = *fc;
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new ConvLSTMPluginV2(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(ConvLSTMPluginV2Creator);
#endif //__CONV_LSTM_HPP__