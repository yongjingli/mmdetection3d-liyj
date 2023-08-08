/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12
 * Use the onnx operator "DecodeNMS" for operator decode & nms

 */

#ifndef __CONFIDENCE_FILTER_HPP__
#define __CONFIDENCE_FILTER_HPP__

#include <math.h>
#include "common.h"

//For Multi-ConfidenceFilter
#define MAX_OFFSET_NUM 16
#define BEV_POOLING_SIZE 5

using namespace nvinfer1;
extern "C" int ConfidenceFilterForward(int batchSize, const void *const *inputs, void *const*outputs, void *worksapce,
                                       cudaStream_t stream, int *params, int *params_int_pointer[5], std::vector<int> &_conf_offsets,
                                       std::vector<Dims> &_input_dims, const void **_d_input_bufs, const DataType mInDataType, const TensorFormat mInTensorFormat);
//Class for Filter, support single input or fpn input
class ConfidenceFilterV2 : public IPluginV2IOExt
{
public:
  ConfidenceFilterV2(const PluginFieldCollection &fc) { (void)fc; }

  ConfidenceFilterV2(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    _max_num = read<int>(d);
    _conf_offset = read<int>(d);
    _conf_thresh = read<float>(d);
    _mode = read<int>(d);
    for (int i = 0; i < 3; ++i)
    {
      _fpn_shape[i] = read<int>(d);
    }
    _ch = _fpn_shape[0];
    _fpn_type_num = read<int>(d);
    _fpn_type.resize(_fpn_type_num);
    for (int i = 0; i < _fpn_type_num; ++i)
    {
      _fpn_type[i] = read<int>(d);
    }
    _conf_offsets_num = read<int>(d);
    _conf_offsets.resize(_conf_offsets_num);
    for (int i = 0; i < _conf_offsets_num; ++i)
    {
      _conf_offsets[i] = read<int>(d);
    }
    _conf_thresholds.resize(_conf_offsets_num);
    for (int i = 0; i < _conf_offsets_num; ++i)
    {
      _conf_thresholds[i] = read<float>(d);
    }
    _input_num = read<int>(d);
    _input_dims.resize(_input_num);
    for (int i = 0; i < _input_num; ++i)
    {
      _input_dims[i] = read<Dims>(d);
    }

    if (d + sizeof(int) * 4 <= a + length) { // support fp16/Int8 format, from v1.0.5std
      mInDataType = (DataType)read<int>(d);
      mOutDataType = (DataType)read<int>(d);
      mInTensorFormat = (TensorFormat)read<int>(d);
      mOutTensorFormat = (TensorFormat)read<int>(d);
    }

    if (d != a + length) {
      DPRINTF(1, "ConfidenceFilterV2 init error! length = %zu\n", length);
    }
  }
  ConfidenceFilterV2(int max_num, std::vector<int> conf_offsets, std::vector<float> conf_threshs, std::string mode,
                     std::vector<int> fpn_shape, std::vector<int> fpn_type)
      : _max_num(max_num)
  {
    _mode = 0;
    if (mode.find("sigmo") != mode.npos) {
      _mode = 1; // sigmoid , for mod/sod, last layer
    }
    if (mode.find("fpn_sum") != mode.npos) {
      _mode |= 0x10; // fpn_sum , for rsm/kptl, replace Channel2Spacis.
    }
    if (mode.find("fpn_concat") != mode.npos) {
      _mode |= 0x20; // fpn_concat, for  mod/sod, replace Channel2Spacis.
    }
    if (mode.find("fpn_nopad_sum") != mode.npos) {
      _mode |= 0x30; // fpn_nopad_sum , for rsm/kptl with zero left/top pad.
    }

    std::copy(fpn_shape.begin(), fpn_shape.end(), _fpn_shape);
    _fpn_type = fpn_type;
    _fpn_type_num = _fpn_type.size();
    _ch = _fpn_shape[0];
    _conf_offsets.clear();
    _conf_thresholds.clear();
    //for bev model
    _conf_offsets = conf_offsets;
    _conf_thresholds = conf_threshs;
    _conf_offsets_num = _conf_offsets.size();
    if( conf_offsets.size() > 0 ){
      _conf_offset = conf_offsets[0];
    }
    if( conf_threshs.size() > 0 ){
      _conf_thresh = conf_threshs[0];
    }    
    DPRINTF(1, "Multi-ConfidenceFilter  create (%d,%d,%.2f,0x%x,%d) _fpn_type%zu\n", _max_num,
            _conf_offset, _conf_thresh, _mode, _ch, _fpn_type.size());
    DPRINTF(1, "Multi-ConfidenceFilter offset = %d :%d :%d \n", _conf_offsets[0], _conf_offsets[1], _conf_offsets[2]);
    float threshold = (_mode & 0x01) ? logf(_conf_thresh / (1 - _conf_thresh)) : _conf_thresh;
    DPRINTF(1, "Multi-ConfidenceFilter inner threshold = %f :%f :%f \n", _conf_thresholds[0], _conf_thresholds[1], _conf_thresholds[2]);
  }

  ~ConfidenceFilterV2() {}
  ConfidenceFilterV2() {}

private:
  int _max_num = 128;   // max number of output
  int _conf_offset = 0; // confidence offset of input
  float _conf_thresh = 0.5f;
  int _mode = 0;           // 0: no sigmod, 1: Softmax. >0x10 with FPN mode
  int _fpn_shape[3] = {0}; // fpn shape: ch, fpn_h, fpn_w
  int _fpn_type_num;
  std::vector<int> _fpn_type; // type of input, 1: Reshape, 3: Channel2Spatial, empty: unknow
  int _conf_offsets_num;
  std::vector<int> _conf_offsets;      // for BEV model
  std::vector<float> _conf_thresholds; // for BEV model
  int _input_num;
  std::vector<Dims> _input_dims;

  bool _initialized = false;
  // runtime only
  int _ch = 0;
  // 0x10: FPNSum(rsm/tl,pad 1,3), 0x20: FPNConcat(mod/sod), 0x30:FPNSum(rsm/tl,nopad)
  int _fpn_mode = 0;
  int *_d_input_dims = nullptr;
  int *_d_input_size = nullptr;
  unsigned char *_d_host_buffer = nullptr; // cudaMallocHost, pinged memory, for combined paramters.
  const void **_d_input_bufs = nullptr;   // Managed Memory
  int *_d_count = nullptr;
  int *_d_conf_offsets = nullptr;      //for BEV model
  float *_d_conf_thresholds = nullptr; //for BEV model
  std::string mNamespace;
  const int _max_batch_size = 4;

  DataType mInDataType = DataType::kFLOAT;
  DataType mOutDataType = DataType::kFLOAT;
  TensorFormat mInTensorFormat = TensorFormat::kLINEAR;
  TensorFormat mOutTensorFormat = TensorFormat::kLINEAR;

public:
  size_t getSerializationSize() const  noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(int);                       //_max_num
    serializationSize += sizeof(int);                       //_conf_offset
    serializationSize += sizeof(float);                     //_conf_thresh
    serializationSize += sizeof(int);                       //_mode
    serializationSize += sizeof(int) * 3;                   //_fpn_shape
    serializationSize += sizeof(int);                       //_fpn_type_num
    serializationSize += sizeof(int) * _fpn_type_num;       //_fpn_type
    serializationSize += sizeof(int);                       //_conf_offsets_num
    serializationSize += sizeof(int) * _conf_offsets_num;   //_conf_offsets
    serializationSize += sizeof(float) * _conf_offsets_num; //_conf_thresholds
    serializationSize += sizeof(int);                       //_input_num
    serializationSize += sizeof(Dims) * _input_num;         //_input_dims
    serializationSize += sizeof(int);                       //mInDataType
    serializationSize += sizeof(int);                       //mOutDataType
    serializationSize += sizeof(int);                       //mInTensorFormat
    serializationSize += sizeof(int);                       //mOutTensorFormat
    return serializationSize;
  }

  void serialize(void *buffer) const noexcept override
  {
    char *d = static_cast<char *>(buffer);
    const char *const a = d;
    write(d, _max_num);
    write(d, _conf_offset);
    write(d, _conf_thresh);
    write(d, _mode);
    for (int i = 0; i < 3; ++i)
    {
      write(d, _fpn_shape[i]);
    }

    write(d, _fpn_type_num);
    for (int i = 0; i < _fpn_type_num; ++i)
    {
      write(d, _fpn_type[i]);
    }
    write(d, _conf_offsets_num);
    for (int i = 0; i < _conf_offsets_num; ++i)
    {
      write(d, _conf_offsets[i]);
    }
    for (int i = 0; i < _conf_offsets_num; ++i)
    {
      write(d, _conf_thresholds[i]);
    }
    write(d, _input_num);
    for (int i = 0; i < _input_num; ++i)
    {
      write(d, _input_dims[i]);
    }
    write(d, (int)mInDataType);
    write(d, (int)mOutDataType);
    write(d, (int)mInTensorFormat);
    write(d, (int)mOutTensorFormat);
    if (d != a + getSerializationSize())
      DPRINTF(1, "ConfidencePluginV2 serialize error!\n");
  }

  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    assert(in && nbInput >= 1);
    assert(out && nbOutput == 1);
    // assert(in[0].type == out[0].type);
    assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
    _input_num = nbInput;
    _input_dims.resize(_input_num);
    for (int i = 0; i < _input_num; i++)
    {
      _input_dims[i] = in[i].dims;
    }

    mInDataType = in[0].type;
    mOutDataType = out[0].type;
    mInTensorFormat = in[0].format;
    mOutTensorFormat = out[0].format;

    DPRINTF(1, "Confidence Filter configType: In=%d Out=%d num: In=%d Out=%d format: In=%d, Out=%d\n", (int)in[0].type, (int)out[0].type, nbInput, nbOutput, (int)mInTensorFormat, (int)mOutTensorFormat);
  }

public:
  int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace, cudaStream_t stream) noexcept override
  {
    if (TRT_DEBUGLEVEL == -1)
    {
      printf("Skip ConfidenceFilter::enqueue!!\n");
      return 0;
    }
    int params[11];
    params[0] = _max_num;
    params[1] = _mode;
    params[2] = _input_num;
    params[3] = _ch;
    params[4] = _max_batch_size;
    params[5] = _fpn_mode;
    params[6] = _conf_offset;
    params[7] = int(_conf_thresh * 100);
    params[8] = _fpn_shape[0];
    params[9] = _fpn_shape[1];
    params[10] = _fpn_shape[2];

    int *params_int_pointer[5];
    params_int_pointer[0] = _d_input_dims;
    params_int_pointer[1] = _d_input_size;
    params_int_pointer[2] = _d_count;
    params_int_pointer[3] = _d_conf_offsets; //for BEV model
    params_int_pointer[4] = (int *)_d_conf_thresholds;
    return ConfidenceFilterForward(batchSize, inputs, outputs, workspace, stream, params, params_int_pointer,
                                   _conf_offsets, _input_dims, _d_input_bufs, mInDataType, mInTensorFormat);
  }
  int getNbOutputs() const noexcept override { return 1; }
  Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override
  {
    if (_input_dims.size() == 0)
    { // not set yet

      _input_dims.assign(inputDims, inputDims + nbInputDims);
      for (int i = 0; i < nbInputDims; i++)
      {
        DPRINTF(1, "Input shape=%d(%d,%d)\n", inputDims[i].nbDims, inputDims[i].d[0], inputDims[i].d[1]);
      }
      _input_num = _input_dims.size();
    }
    if (0 == _ch)
    { // not set yet
      if (3 == _input_dims[0].nbDims)
      { // CHW
        _ch = _input_dims[0].d[0];
      }
      else
      {
        _ch = _input_dims[0].d[1];
      }
    }
    nvinfer1::Dims output = Dims2(_max_num, _ch + 1);
    DPRINTF(1, "Output shape=%d(%d,%d)\n", output.nbDims, output.d[0], output.d[1]);
    return output;
  }

  int initialize() noexcept override
  {
    if (_initialized)
    {
      return 0;
    }

    _input_num = _input_dims.size();
    _fpn_mode = _mode & 0xF0;
    DPRINTF(2, "_fpn_mode is %d\n", _fpn_mode);
    // if (0 != _fpn_mode)
    // {            
      int algn_input_num = ( _input_num + 3 ) / 4 * 4; // aligned offset for cuda address                                                // FPN mode
      int size = _max_batch_size * algn_input_num * sizeof(void *); // input buffer GPU ptr.
      size += algn_input_num * sizeof(int4);                         // w, h, ratio, num,
      size += algn_input_num * sizeof(int);                          // input size
      size += MAX_OFFSET_NUM * sizeof(int);                      // //only used in BEV model
      size += MAX_OFFSET_NUM * sizeof(float);
      cudaHostAlloc(&_d_host_buffer, size, cudaHostAllocMapped);
      _d_input_bufs = (const void **)_d_host_buffer;
      _d_input_dims = (int *)(_d_input_bufs + _max_batch_size * algn_input_num);
      _d_input_size = _d_input_dims + algn_input_num * sizeof(int4) / sizeof(int);
      _d_conf_offsets = _d_input_size + algn_input_num;
      _d_conf_thresholds = (float *)_d_conf_offsets + MAX_OFFSET_NUM;
      DPRINTF(2, "End of _d_host_buffer %p = %p\n", _d_host_buffer + size, _d_conf_thresholds + MAX_OFFSET_NUM);

      if (_conf_thresholds.size() > 0 ){
        DPRINTF(2, "Mulit-ConfidenceFilter offset = (");
        for (size_t i = 0; i < _conf_offsets.size(); i++){
          _d_conf_offsets[i]=_conf_offsets[i];
          DPRINTF(2, "%d : ", _conf_offsets[i]); 
        } 
        DPRINTF(2, ")\nMulit-ConfidenceFilter inner threshold = (");
        for (size_t i = 0; i < _conf_thresholds.size(); i++){
          _d_conf_thresholds[i]= (_mode & 0x01) ? logf(_conf_thresholds[i]/(1-_conf_thresholds[i])) : _conf_thresholds[i];
          DPRINTF(2, "%.3f : ", _d_conf_thresholds[i]); 
        }
        _conf_thresh = _conf_thresholds[0]; // _conf_thresh as same as _conf_thresholds[0]
        DPRINTF(2, ") _conf_thresh=%.3f\n", _conf_thresh); 
      }

      if (_conf_offsets.size() == 3)
      {
        DPRINTF(1, "BEV ConfidenceFilter offset = %d :%d :%d \n", _conf_offsets[0], _conf_offsets[1], _conf_offsets[2]);
        DPRINTF(1, "BEV ConfidenceFilter inner threshold = %f :%f :%f \n", _conf_thresholds[0], _conf_thresholds[1], _conf_thresholds[2]);
      }
      int4 *d_input_dim = (int4 *)_d_input_dims;
      int offset = 0;
      for (int i = 0; i < _input_num; i++)
      {
        auto const &input_dim = _input_dims[i];
        d_input_dim->x = input_dim.d[2]; // w
        d_input_dim->y = input_dim.d[1]; // h
        if (0x20 == _fpn_mode)
        {                                        // FPNConcat
          d_input_dim->z = input_dim.d[0] / _ch; // ratio, reshape
          d_input_dim->w = offset;               // offset
          int num = d_input_dim->z * d_input_dim->y * d_input_dim->x;
          offset += num;
          if ((_fpn_type.size() > (size_t)i && _fpn_type[i] == 3) || (_fpn_type.size() <= (size_t)i && d_input_dim->z == 4))
          {
            d_input_dim->z = -d_input_dim->z; // -ratio, channel2spatial
          }
        }
        else
        {                                               // FpnSum, pad 1,3 or pad 0,0
          d_input_dim->z = sqrtf(input_dim.d[0] / _ch); // ratio
          d_input_dim->w = d_input_dim->x * d_input_dim->z * d_input_dim->y * d_input_dim->z;
        }

        DPRINTF(2, "FPN[%d] in(%d,%d,%d) -> dev(%d,%d,%d,%d)\n", i,
                input_dim.d[0], input_dim.d[1], input_dim.d[2],
                d_input_dim->x, d_input_dim->y, d_input_dim->z, d_input_dim->w);
        d_input_dim++;
      }
    // }
    cudaMalloc(&_d_count, _max_batch_size * _input_num * sizeof(int) * (MAX_OFFSET_NUM + 1));
    _initialized = true;
    return 0;
  }

  void terminate() noexcept override
  {
    if (!_initialized)
    {
      return;
    }

    if (_d_host_buffer)
    {
      cudaFreeHost(_d_host_buffer);
    }
    if (_d_count)
    {
      cudaFree(_d_count);
    }
    _initialized = false;
  }
  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbInputs >= 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= ((inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF));
    if (pos < nbInputs) {
      condition &= inOut[pos].type == inOut[0].type;
    } else {
      condition &= inOut[pos].type == DataType::kFLOAT;
    }
    return condition;
  }
  DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
  {
    assert(inputTypes && nbInputs >= 1);
    (void)index;
    return inputTypes[0];
  }
  const char *getPluginType() const noexcept override { return "ConfidenceFilter"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  void destroy() noexcept override { delete this; }
  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new ConfidenceFilterV2(*this);
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

class ConfidenceFilterV2Creator : public IPluginCreator
{
public:
  ConfidenceFilterV2Creator(){
    mPluginAttributes.emplace_back(PluginField("max_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("conf_offset", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("conf_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("fpn_shape", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("fpn_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kCHAR, 1));
    DPRINTF(2, "ConfidenceFilterV2Creator add mPluginAttributes.\n");
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }
  
  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    DPRINTF(2, "ConfidenceFilterV2Creator mPluginName=%s\n", name);
    mPluginName = name;  
    const PluginField* fields = fc->fields;
    std::vector<int> conf_offsets{0};
    std::vector<float> conf_threshs{0.5};
    std::vector<int> fpn_shape{16,240,480};
    std::vector<int> fpn_type{1};
    int max_num = 128;
    std::string mode;
    for (auto i = 0; i < fc->nbFields; ++i)
    {  
      const char* attrName = fields[i].name;
      DPRINTF(2, "ConfidenceFilterV2Creator attrName=%s len=%d\n", attrName, fields[i].length);
      if (!strcmp(attrName, "conf_offset")) {  
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          conf_offsets.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              conf_offsets[j] = *aR;
              DPRINTF(2, "conf_offsets[%d]=%d\n", j, conf_offsets[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "conf_threshold")) {
        assert(fields[i].type == PluginFieldType::kFLOAT32);
        const int32_t size = fields[i].length / sizeof(float);
        if (size > 0) {
          conf_threshs.resize(size);
          const auto* aR = static_cast<const float*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              conf_threshs[j] = *aR;
              DPRINTF(2, "conf_threshs[%d]=%f\n", j, conf_threshs[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "fpn_shape")) {
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          fpn_shape.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              fpn_shape[j] = *aR;
              DPRINTF(2, "fpn_shape[%d]=%d\n", j, fpn_shape[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "fpn_type")) {
        //assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          fpn_type.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              fpn_type[j] = *aR;
              DPRINTF(2, "fpn_type[%d]=%d\n", j, fpn_type[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "max_num")){
        assert(fields[i].type == PluginFieldType::kINT32);
        max_num = *static_cast<const int*>(fields[i].data);
      } else if (!strcmp(attrName, "mode")){
        assert(fields[i].type == PluginFieldType::kCHAR);
        mode.insert(0, static_cast<const char*>(fields[i].data), fields[i].length);
      }
    }
    DPRINTF(2, "ConfidenceFilterV2Creator max_num=%d,mode=%s\n", max_num, mode.c_str());  
    auto plugin = new ConfidenceFilterV2(max_num, conf_offsets, conf_threshs, mode, fpn_shape, fpn_type);
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new ConfidenceFilterV2(serialData, serialLength);
    mPluginName = name;
    return plugin;
  }

  const char *getPluginName() const noexcept override { return "ConfidenceFilter"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
  std::string mNamespace;
  std::string mPluginName;
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(ConfidenceFilterV2Creator);
#endif //__CONFIDENCE_FILTER_HPP__
