#ifndef __GEMVIN8_HPP__
#define __GEMVIN8_HPP__

#include "common.h"
using namespace nvinfer1;
using namespace std;
#define KTYPE4 char4 // float4
#define KTYPE char   // float
inline std::vector<int> partsort_indexes(const float v[], int vsize,
                                         int partsort = 0)
{
    // initialize original index locations
    std::vector<int> idx(vsize);
    for (int i = 0; i < vsize; ++i)
        idx[i] = i;

    // sort indexes based on comparing values in v
    std::partial_sort(idx.begin(), idx.begin() + partsort, idx.end(),
                      [&v](int i1, int i2) { return fabs(v[i1]) > fabs(v[i2]); });
    idx.resize(partsort);

    return idx;
}

inline void calInt8Weight(const float *value, int rows, std::vector<KTYPE> &kernels,
                          std::vector<float> &scales)
{
    char *val = getenv("TRT_GEMV");
    int partNum = 1;
    if (NULL != val)
    {
        partNum = atoi(val);
        DPRINTF(2, "getenv TRT_GEMV partNum = %d\n", partNum);
        if (partNum <= 0)
            return;
    }

    int offset = 0;
    int n = kernels.size() / rows;
    DPRINTF(3, "Gemv calInt8Weight m=%d, n=%d\n", rows, n);
    for (int m = 0; m < rows; m++)
    {
        auto idx = partsort_indexes((value + offset), n, partNum);
        float maxVal = fabs(value[offset + idx[partNum - 1]]);
        float scale = (maxVal + 1e-7) / 127.f;
        DPRINTF(4, "%d Gemv [%f,%f...] scale %f maxVal=%f\n", m, value[offset],
                value[offset + 1], scale, maxVal);
        scales[m] = scale;
        for (int i = 0; i < n; i++, offset++)
        {
            int tmpval = (int)round(value[offset] / scale);
            kernels[offset] =
                (tmpval > 127) ? 127 : ((tmpval < -128) ? -128 : tmpval);
        }
    }
}
extern "C" int GemvInt8PluginV2Forward(int batchSize, const void *const *inputs,
                                       void *const *outputs, void *workspace,
                                       cudaStream_t stream, int m, int qn, KTYPE *_d_kernel,
                                       float *_d_bias, float *_d_scale);
class GemvInt8PluginV2 : public IPluginV2IOExt
{
public:
    GemvInt8PluginV2(const PluginFieldCollection *fc) {  
      const PluginField* fields = fc->fields;
      const float* kernel = nullptr; 
      nvinfer1::Dims shape;
      for (auto i = 0; i < fc->nbFields; ++i)
      {  
        const char* attrName = fields[i].name;
        DPRINTF(2, "GemvInt8PluginV2 attrName=%s len=%d\n", attrName, fields[i].length);
        if (!strcmp(attrName, "weight")) {  
          assert(fields[i].type == PluginFieldType::kFLOAT32);
          const int32_t size = fields[i].length;
          if (size > 0) {
            _h_kernel.resize(size);
            kernel = static_cast<const float*>(fields[i].data);
          }
        } else if (!strcmp(attrName, "bias")) {  
          assert(fields[i].type == PluginFieldType::kFLOAT32);
          const int32_t size = fields[i].length;
          if (size > 0) {
            _h_bias.resize(size);
            const auto* aR = static_cast<const float*>(fields[i].data);
            for (auto j = 0; j < size; j++) {
                _h_bias[j] = *aR;
                aR++;
            }
          }
        } else if (!strcmp(attrName, "shape")) {  
          assert(fields[i].type == PluginFieldType::kINT32);
          const int32_t size = fields[i].length;
          if (size > 0) {
            shape.nbDims = size;
            const auto* aR = static_cast<const int*>(fields[i].data);
            for (auto j = 0; j < size; j++) {
                shape.d[j] = *aR;
                DPRINTF(3, "_shape[%d]=%d\n", j, shape.d[j]);
                aR++;
            }
          }
        }
      }
      
      _nrow = _h_bias.size();
      _bias_num = _nrow;      
      _kernel_num = _h_kernel.size();
      _scale_num = _nrow;
      _h_scale.resize(_scale_num, 0.0f);
      if( kernel ){ 
        calInt8Weight(kernel, _nrow, _h_kernel, _h_scale);
      } else {
        DPRINTF(1, "Error: GemvInt8PluginV2 has not weight\n");
      }
      _output_dims = DimsHW(_nrow, 1);
      if( shape.nbDims == 3 && shape.d[2] > 1 ) {
        _output_dims.d[0] /= shape.d[2];
        _output_dims.d[1]  = shape.d[2];
      }      
    }
    
    GemvInt8PluginV2(int rows, nvinfer1::Weights const &kernel,
                     nvinfer1::Weights const &bias, const std::vector<int> &shape)
        : _nrow(rows) {
        assert(rows == bias.count);
        if (bias.type == nvinfer1::DataType::kFLOAT)
        {
            _h_bias.assign((float *)bias.values, (float *)bias.values + bias.count);
            _h_scale.assign((float *)bias.values, (float *)bias.values + bias.count);
        }
        else
        {
            throw std::runtime_error("Unsupported bias dtype");
        }

        if (kernel.type == nvinfer1::DataType::kFLOAT)
        {
            _h_kernel.assign((KTYPE *)kernel.values,
                             (KTYPE *)kernel.values + kernel.count);
            calInt8Weight((float *)kernel.values, rows, _h_kernel, _h_scale);
        }
        else
        {
            throw std::runtime_error("Unsupported kernel dtype");
        }
        _kernel_num = _h_kernel.size();
        _bias_num = _h_scale.size();
        _scale_num = _h_scale.size();
        if (shape.size() == 1 && shape[0] == 0) {
            _output_dims = DimsHW(_nrow, 1);
            _output_dims.nbDims = 1;
        } else {
            _output_dims = DimsHW(_nrow, shape[2]);
            if( shape.size() == 3 && shape[2] > 1 ) {
                _output_dims.d[0] /= shape[2];
                _output_dims.d[1]  = shape[2];
            }    
        }
    }

    GemvInt8PluginV2(const void *data, size_t length)
    {
        const char *d = static_cast<const char *>(data);
        const char *const a = d;
        _nrow = read<int>(d);
        _kernel_num = read<int>(d);
        _h_kernel.resize(_kernel_num);
        for (int i = 0; i < _kernel_num; ++i)
        {
            _h_kernel[i] = read<KTYPE>(d);
        }
        _bias_num = read<int>(d);
        _h_bias.resize(_bias_num);
        for (int i = 0; i < _bias_num; ++i)
        {
            _h_bias[i] = read<float>(d);
        }
        _scale_num = read<int>(d);
        _h_scale.resize(_scale_num);
        for (int i = 0; i < _scale_num; ++i)
        {
            _h_scale[i] = read<float>(d);
        }
        _input_dims = read<Dims>(d);
        _output_dims = read<Dims>(d);        
        DPRINTF(1, "deserialize GEMV Input=%d(%d,%d,%d)\n", _input_dims.nbDims,
                _input_dims.d[0], _input_dims.d[1], _input_dims.d[2]);
        if (d != a + length)
            DPRINTF(1, "GemvInt8PluginV2 init error!\n");
    }

    GemvInt8PluginV2() {}

    ~GemvInt8PluginV2() {}

public:
    int enqueue(int batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                cudaStream_t stream) noexcept override
    {
        assert(_initialized);
        if (TRT_DEBUGLEVEL == -1)
        {
            printf("Skip GemvInt8PluginV2::enqueue!!\n");
            return 0;
        }
        //nvinfer1::Dims input_dims = this->getInputDims(0);
        int qn = _input_dims.d[0]; // quarter of n 
        for( int i=1; i<_input_dims.nbDims; i++ ){
          qn *= _input_dims.d[i];
        }
        qn = std::min(_ncol, qn)  / 4 ; // No more then kernel col. As float4
        
        int m = _nrow;
        return GemvInt8PluginV2Forward(batchSize, inputs, outputs, workspace, stream, m, qn, _d_kernel,
                                       _d_bias, _d_scale);
    }
    int getNbOutputs() const noexcept override { return 1; }
    Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override
    {
        assert(index == 0);
        _input_dims = inputs[0];
        DPRINTF(1, "GEMV Input=%d(%d,%d,%d)\n", _input_dims.nbDims,
                _input_dims.d[0], _input_dims.d[1], _input_dims.d[2]);
        if (_output_dims.nbDims == 2) {
            DPRINTF(1, "GEMV Output=%d(%d,%d)\n", _output_dims.nbDims, _output_dims.d[0], _output_dims.d[1]);
        } else {
            DPRINTF(1, "GEMV Output=%d(%d)\n", _output_dims.nbDims, _output_dims.d[0]);
        }
        return _output_dims;
    }

    int initialize() noexcept override
    {
        if (_initialized)
        {
            return 0;
        }
        _ncol = _kernel_num / _nrow;
        DPRINTF(2, "GemvInt8Plugin initialize start,_h_kernel=%d(%dx%d)\n", _kernel_num, _nrow, _ncol);

        size_t nkernel_bytes = _kernel_num * sizeof(KTYPE);
        size_t nbias_bytes = _nrow * sizeof(float);
        CHECK_CUDA(cudaMalloc((void **)&_d_kernel, nkernel_bytes));
        CHECK_CUDA(cudaMalloc((void **)&_d_bias, nbias_bytes));
        CHECK_CUDA(cudaMalloc((void **)&_d_scale, nbias_bytes));
        cudaStream_t stream = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CHECK_CUDA(cudaMemcpyAsync(_d_kernel, _h_kernel.data(), nkernel_bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(_d_bias, _h_bias.data(), nbias_bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(_d_scale, _h_scale.data(), nbias_bytes, cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        _initialized = true;
        DPRINTF(2, " GemvInt8Plugin initialize done\n");
        return 0;
    }

    void terminate() noexcept override
    {
        if (!_initialized)
        {
            return;
        }
        DPRINTF(2, "GemvInt8Plugin terminate start\n");
        CHECK_CUDA(cudaFree(_d_scale));
        CHECK_CUDA(cudaFree(_d_bias));
        CHECK_CUDA(cudaFree(_d_kernel));
        DPRINTF(2, "GemvInt8Plugin terminate done\n");
        _initialized = false;
    }
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override
    {
        auto const &input0_dims = _input_dims;
        int nsize = maxBatchSize * input0_dims.d[0] * sizeof(float);
        DPRINTF(2, "GemvInt8Plugin getWorkspaceSize = %d\n", nsize);
        return nsize;
    }
    size_t getSerializationSize() const noexcept override
    {
        size_t serializationSize = 0;
        serializationSize += sizeof(int);                 //_nrow
        serializationSize += sizeof(int);                 //_kernel_num
        serializationSize += sizeof(KTYPE) * _kernel_num; //_h_kernel
        serializationSize += sizeof(int);                 //_bias_num
        serializationSize += sizeof(float) * _bias_num;   //_h_bias
        serializationSize += sizeof(int);                 //_scale_num
        serializationSize += sizeof(float) * _scale_num;  //_h_scale
        serializationSize += sizeof(Dims);                //_input_dims
        serializationSize += sizeof(Dims);                //_output_dims        
        return serializationSize;
    }

    void serialize(void *buffer) const noexcept override
    {
        char *d = static_cast<char *>(buffer);
        const char *const a = d;
        write(d, _nrow);
        write(d, _kernel_num);
        for (int i = 0; i < _kernel_num; ++i)
        {
            write(d, _h_kernel[i]);
        }
        write(d, _bias_num);
        for (int i = 0; i < _bias_num; ++i)
        {
            write(d, _h_bias[i]);
        }
        write(d, _scale_num);
        for (int i = 0; i < _scale_num; ++i)
        {
            write(d, _h_scale[i]);
        }
        write(d, _input_dims);
        write(d, _output_dims);
        if (d != a + getSerializationSize())
            DPRINTF(1, "GemvInt8PluginV2 serialize error!\n");
    }

    void configurePlugin(const PluginTensorDesc *in, int nbInput,
                         const PluginTensorDesc *out, int nbOutput) noexcept override
    {
        //assert(in && nbInput == 1);
        assert(out && nbOutput == 1);
        assert(in[0].type == out[0].type);
        assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
        //_input_dims = in[0].dims;
        DPRINTF(1, "configType: In=%d Out=%d \n", (int)in[0].type, (int)out[0].type);
    }
    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override
    {
        assert(nbOutputs == 1 && pos < nbInputs + nbOutputs);
        bool condition = inOut[pos].format == TensorFormat::kLINEAR;

        condition &= inOut[pos].type == DataType::kFLOAT;
        condition &= inOut[pos].type == inOut[0].type;
        return condition;
    }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        //assert(inputTypes && nbInputs == 1);
        (void)index;
        return inputTypes[0];
    }
    const char *getPluginType() const noexcept override { return "GemvInt8"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    void destroy() noexcept override { delete this; }

    IPluginV2Ext *clone() const noexcept override
    {
        auto *plugin = new GemvInt8PluginV2(*this);
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
    int _nrow;
    int _ncol; // = _kernel_num / _nrow
    int _kernel_num;
    int _bias_num;
    int _scale_num;
    std::vector<KTYPE> _h_kernel;
    std::vector<float> _h_bias;
    std::vector<float> _h_scale;    
    Dims _input_dims = {0,{0,}};
    Dims _output_dims = {0,{0,}};
    KTYPE *_d_kernel;
    float *_d_bias;
    float *_d_scale;
    std::string mNamespace;
    bool _initialized = {false};
};

class GemvInt8PluginV2Creator : public IPluginCreator
{
public:
  GemvInt8PluginV2Creator(){
    mPluginAttributes.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("shape", nullptr, PluginFieldType::kINT32, 1));
      
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }
  
    const char *getPluginName() const noexcept override { return "GemvInt8"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        auto plugin = new GemvInt8PluginV2(fc);
        mPluginName = name;
        return plugin;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        auto plugin = new GemvInt8PluginV2(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(GemvInt8PluginV2Creator);
#endif //__GEMVIN8_HPP__
