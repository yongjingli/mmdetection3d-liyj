#ifndef __YHATS_FILTER_HPP__
#define __YHATS_FILTER_HPP__

#include <math.h>
#include <vector>

#include "common.h"

using namespace nvinfer1;

extern "C" int YhatsFilterForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                                    std::vector<int> inDims, int chOffset, float threOffset, cudaStream_t stream);

class YhatsFilter : public IPluginV2IOExt
{

//Constructor
public:
    YhatsFilter(const PluginFieldCollection &fc) { (void)fc; }

    //This consturctor is used for desereialezed
    YhatsFilter(const void *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        const char *const a = d;
        _ic       = read<int>(d);
        _ih       = read<int>(d);
        _iw       = read<int>(d);
        _radius   = read<int>(d);
        _chOffset = read<int>(d);

        _fOffset  = read<float>(d);

        if (d != a + length)
            DPRINTF(1, "YhatsFilter init error!\n");

    }

    //This constructor is used for BUILDIN_OP_IMPORTER
    YhatsFilter(int channel, int height, int width, int radius, int chOffset, float fOffset):
    _ic(channel), _ih(height), _iw(width), _radius(radius), _chOffset(chOffset), _fOffset(fOffset) {}


//PLUGIN CONFIGURATION
public:
    int getNbOutputs() const noexcept override { return 1; }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override {
        assert(inputTypes && nbInputs == 1);
        (void)index;
        return inputTypes[0];
    }

    Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override {
        nvinfer1::Dims output = Dims2(1, _iw + (_ic + 1) * (_radius * 2 + 1) * _iw);  //Where to illustrate batch size?
        DPRINTF(1, "Output shape=%d(%d,%d)\n", output.nbDims, output.d[0], output.d[1]);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override {
        assert(nbInputs  >= 1);
        assert(nbOutputs == 1);
        assert(pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= inOut[pos].type == DataType::kFLOAT;
        return condition;
    }
  
    void configurePlugin(const PluginTensorDesc *in,  int nbInput,
                         const PluginTensorDesc *out, int nbOutput) noexcept override {
        assert(in  && nbInput  >= 1);
        assert(out && nbOutput == 1);
        for(int i = 0; i < nbInput; ++i) {
            assert(in[i].type   == out[0].type);
            assert(in[i].format == out[0].format);
        }
        assert(out[0].format == TensorFormat::kLINEAR);
        assert(out[0].type   == DataType::kFLOAT);
        
        DPRINTF(2, "Yhats configType: In=%d Out=%d num: In=%d Out=%d \n", (int)in[0].type, (int)out[0].type, nbInput, nbOutput);
    }

//PUGIN CONSISTENCY CHECKING
public:
    const char *getPluginType() const noexcept override { return "YhatsFilter"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

    const char *getPluginNamespace() const noexcept override { return mNamespace.data(); }

//PLUGIN SERIALIZATION
public:

    size_t getSerializationSize() const noexcept override {
        size_t serializationSize = 0;
        serializationSize += sizeof(int);    //_ic
        serializationSize += sizeof(int);    //_ih
        serializationSize += sizeof(int);    //_iw
        serializationSize += sizeof(int);    //_radius
        serializationSize += sizeof(int);    //_chOffset
        serializationSize += sizeof(float);  //_fOffset

        return serializationSize;
    }

    void serialize(void *buffer) const noexcept override {
        char *d = static_cast<char *>(buffer);
        const char *const a = d;
        write(d, _ic);
        write(d, _ih);
        write(d, _iw);
        write(d, _radius);
        write(d, _chOffset);
        write(d, _fOffset);

        if (d != a + getSerializationSize())
            DPRINTF(1, "YhatsFilter serialize error!\n");

    }

//PLUGIN EXECUTION
public:
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

    void terminate() noexcept override {}

    int initialize() noexcept override { return 0; }

    int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace, cudaStream_t stream) noexcept override {
        std::vector<int> inDims(4, 0);
        inDims[0] = _ic;
        inDims[1] = _ih;
        inDims[2] = _iw;
        inDims[3] = _radius;
        return YhatsFilterForward(batchSize, inputs, outputs, workspace, inDims, _chOffset, _fOffset, stream);
    }

    void destroy() noexcept override { delete this; }

    IPluginV2Ext *clone() const noexcept override {
        auto *plugin = new YhatsFilter(*this);
        return plugin;
    }

public:
    
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }


private:
    int  _radius   = 5;
    int  _iw       = 320;
    int  _ih       = 128;
    int  _ic       = 8;
    int  _chOffset = 0;

    float _fOffset  = 112.0f;

    std::string mNamespace;
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

class YhatsFilterCreator : public IPluginCreator
{
public:
    const char *getPluginName() const noexcept override { return "YhatsFilter"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    const PluginFieldCollection *getFieldNames() noexcept override { return &mFieldCollection; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
        auto plugin = new YhatsFilter(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
        auto plugin = new YhatsFilter(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(YhatsFilterCreator);
#endif