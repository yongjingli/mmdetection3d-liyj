#ifndef FUSEDMULTIHEADATTENTION_H_
#define FUSEDMULTIHEADATTENTION_H_

#include <math.h>
#include <vector>

#include "common.h"
#include <cuda_runtime.h>
#include <cuda.h>

using namespace nvinfer1;
using namespace std;

#define NUMBER_INPUT_DIMS  3

int FusedMultiHeadAttentionForward_fp16(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
        float scale, std::vector<std::vector<int>>& inputDims, cudaStream_t stream);

class FusedMultiHeadAttention : public IPluginV2IOExt {
//Constructor
public:
    FusedMultiHeadAttention(const PluginFieldCollection *fc) { 
        const PluginField* fields = fc->fields;
        for (auto i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "scale")) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                _scale = *static_cast<const float*>(fields[i].data);
                DPRINTF(1, "FusedMultiHeadAttention exp scale is=%.6f\n", _scale);
            }
        }
    }

    //This consturctor is used for desereialezed
    FusedMultiHeadAttention(const void *serialData, size_t serialLength) {
        const char * d = static_cast<const char *>(serialData);
        const char * const a = d;
        _scale = read<float>(d);
        _inputDims.resize(NUMBER_INPUT_DIMS);
        for (int i = 0; i < NUMBER_INPUT_DIMS; ++i) {
            _inputDims[i].resize(3);
            for(int j = 0; j < 3; ++j)
                _inputDims[i][j] = read<int>(d);            
        }
        _mDataType = static_cast<DataType>(read<int>(d));
        if (d != a + serialLength)
            DPRINTF(1, "FusedMultiHeadAttention init error!\n");
    }

    //This constructor is used for BUILDIN_OP_IMPORTER
    FusedMultiHeadAttention(float scale) : _scale(scale) {}

    // FusedMultiHeadAttention() = delete;
    // ~FusedMultiHeadAttention() override;

//PLUGIN CONFIGURATION
public:
    int getNbOutputs() const noexcept override { return 1; }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override {
        assert(inputTypes && nbInputs == NUMBER_INPUT_DIMS);

        DPRINTF(1, "FusedMultiHeadAttention input type= %d %d %d %d\n", 
                inputTypes[0], inputTypes[1], inputTypes[2], inputTypes[3]);

        (void)index;
        return inputTypes[0];
    }

    Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override {
        assert(inputDims && nbInputDims == NUMBER_INPUT_DIMS);
        assert(inputDims[0].nbDims == 3 && inputDims[1].nbDims == 3 && inputDims[2].nbDims == 3);
        assert(inputDims[0].d[2] == inputDims[1].d[1] && inputDims[1].d[2] == inputDims[2].d[1]);
        nvinfer1::Dims output = inputDims[0];
        _inputDims.resize(nbInputDims);
        for (int i = 0; i < nbInputDims; ++i) {
            _inputDims[i].resize(inputDims[i].nbDims);
            for(int j = 0; j < inputDims[i].nbDims; ++j) {
                // DPRINTF(1, "FusedMultiHeadAttention inputDims[%d][%d] = %d\n", i, j, inputDims[i].d[j]);
                _inputDims[i][j] = inputDims[i].d[j];    
            }
        }
        DPRINTF(1, "FusedMultiHeadAttention Output shape=%d(%d,%d,%d)\n", output.nbDims, output.d[0], output.d[1],output.d[2]);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override {
        assert(nbInputs  == NUMBER_INPUT_DIMS);
        assert(nbOutputs == 1);
        assert(pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        // condition &= ((inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF));
        condition &= (inOut[pos].type == DataType::kHALF);
        condition &= inOut[pos].type == inOut[0].type;

        return condition;
    }

    void configurePlugin(const PluginTensorDesc *in,  int nbInput,
                         const PluginTensorDesc *out, int nbOutput) noexcept override {
        assert(in  && nbInput  == NUMBER_INPUT_DIMS);
        assert(out && nbOutput == 1);
        assert(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
        for(int i = 0; i < nbInput; ++i) {
            assert(in[i].type   == out[0].type);
            assert(in[i].format == out[0].format);
        }

        _inputDims.resize(nbInput);
        for (int i = 0; i < nbInput; i++) {
            _inputDims[i].resize(in[i].dims.nbDims);
            for (int j=0; j<in[i].dims.nbDims; ++j) {
                _inputDims[i][j] = in[i].dims.d[j];
            }
        }

        _mDataType = in[0].type;
        
        DPRINTF(1, "FusedMultiHeadAttention configType: In=%d Out=%d num: In=%d Out=%d \n", (int)in[0].type, (int)out[0].type, nbInput, nbOutput);
    }

//PLUGIN SERIALIZATION
public:

    size_t getSerializationSize() const noexcept override {
        size_t serializationSize = 0;
        serializationSize += sizeof(float);    //_scale
        serializationSize += _inputDims.size() * _inputDims[0].size() * sizeof(int); //input dimension
        serializationSize += sizeof(static_cast<int>(_mDataType));
        return serializationSize;
    }

    void serialize(void *buffer) const noexcept override {
        char *d = static_cast<char *>(buffer);
        const char *const a = d;
        write(d, _scale);
        for(int i = 0; i < _inputDims.size(); ++i)
            for(int j = 0; j < _inputDims[i].size(); ++j)
            write(d, _inputDims[i][j]);
        write(d, static_cast<int>(_mDataType));

        if (d != a + getSerializationSize())
            DPRINTF(1, "FusedMultiHeadAttention serialize error!\n");
    }

//PUGIN CONSISTENCY CHECKING
public:
    const char *getPluginType() const noexcept override { return "FusedMultiHeadAttention"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(const char *libNamespace) noexcept override { _mNamespace = libNamespace; }

    const char *getPluginNamespace() const noexcept override { return _mNamespace.data(); }

//PLUGIN EXECUTION
public:
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { 
        if (_mDataType == DataType::kHALF) {
            return _inputDims[0][1] * _inputDims[1][2] * sizeof(__half);           
        }
        if (_mDataType == DataType::kINT8) {
            return _inputDims[0][1] * _inputDims[1][2] * sizeof(int8_t);             
        }
        return _inputDims[0][1] * _inputDims[1][2] * sizeof(float); 
    }

    void terminate() noexcept override {
    }

    int initialize() noexcept override { 
        // printf (">>> initialize scale: %f\n", _scale);
        // printf (">>> initialize inputdims: %d %d %d\n", _inputDims[0][0], _inputDims[0][1], _inputDims[0][2]);
        // _scale = 1.0;
        return 0;
    }

    int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace, cudaStream_t stream) noexcept override {
        if (TRT_DEBUGLEVEL == -1) {
            printf("Skip FusedMultiHeadAttention::enqueue!!\n");
            return 0;
        }
        if (_mDataType == DataType::kHALF) 
            return FusedMultiHeadAttentionForward_fp16(batchSize, inputs, outputs, workspace, _scale, _inputDims, stream);

        return 0;
    }

    void destroy() noexcept override { 
        delete this; 
    }

    IPluginV2Ext *clone() const noexcept override {
        auto *plugin = new FusedMultiHeadAttention(*this);
        return plugin;
    }

public:
    
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { 
        return false; 
    }

private:
    float              _scale;

    std::vector<std::vector<int>> _inputDims;

    DataType         _mDataType;

    std::string      _mNamespace;

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

};

class FusedMultiHeadAttentionCreator : public IPluginCreator {
public:
    FusedMultiHeadAttentionCreator() {
        mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
    }
    const char *getPluginName() const noexcept override { 
        return "FusedMultiHeadAttention"; 
    }

    const char *getPluginVersion() const noexcept override { return "1"; }

    const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
        auto plugin = new FusedMultiHeadAttention(fc);
        mPluginName = name;
        return plugin;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
        auto plugin = new FusedMultiHeadAttention(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(FusedMultiHeadAttentionCreator);

#endif