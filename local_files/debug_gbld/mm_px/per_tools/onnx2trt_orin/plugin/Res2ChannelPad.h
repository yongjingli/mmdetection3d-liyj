#ifndef __RES_2_CHANNEL_PAD_H__
#define __RES_2_CHANNEL_PAD_H__

#include <math.h>
#include <vector>

#include "common.h"

using namespace nvinfer1;

extern "C" int Res2ChannelPadForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                            std::vector<uint32_t> inDims, uint32_t rowPadding, uint32_t columnPadding, DataType inDataType, cudaStream_t stream);

class Res2ChannelPad : public IPluginV2IOExt {
public:
    Res2ChannelPad(const PluginFieldCollection &fc) { (void)fc; }

    //This consturctor is used for desereialezed
    Res2ChannelPad(const void *data, size_t length) {
        const char *d = static_cast<const char *>(data);
        const char *const a = d;
        _resolution    = read<uint32_t>(d);
        _rowPadding    = read<uint32_t>(d);
        _columnPadding = read<uint32_t>(d);
        _inChannels    = read<uint32_t>(d);
        _inHeight      = read<uint32_t>(d);
        _inWidth       = read<uint32_t>(d);

        _mInDataType  = static_cast<DataType>(read<int>(d));

        if (d != a + length)
            DPRINTF(1, "Res2ChannelPad init error!\n");

    }


    //This constructor is used for BUILDIN_OP_IMPORTER
    Res2ChannelPad(uint32_t resolution, uint32_t rowPadding, uint32_t columnPadding):
    _resolution(resolution), _rowPadding(rowPadding), _columnPadding(columnPadding){}

//PLUGIN CONFIGURATION
public:
    int getNbOutputs() const noexcept override { return 1; }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override {
        assert(inputTypes && nbInputs == 1);
        (void)index;
        return inputTypes[0];
    }

    Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override {
        _inChannels = inputDims->d[0]; _inHeight = inputDims->d[1]; _inWidth = inputDims->d[2];
        nvinfer1::Dims output = Dims3(_inChannels* _resolution * _resolution, 
                                        _inHeight / _resolution + _rowPadding, 
                                        _inWidth  / _resolution + _columnPadding);
        DPRINTF(1, "Res2ChannelPad Output shape=%d(%d,%d,%d)\n", output.nbDims, output.d[0], output.d[1], output.d[2]);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override {
        assert(nbInputs  >= 1);
        assert(nbOutputs == 1);
        assert(pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition &= (inOut[pos].type == DataType::kINT8 || inOut[pos].type == DataType::kFLOAT);
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
        assert(out[0].type   == DataType::kINT8 || out[0].type == DataType::kFLOAT);

        _mInDataType  = in[0].type;
        
        DPRINTF(2, "Res2ChannelPad configType: In=%d Out=%d num: In=%d Out=%d \n", (int)in[0].type, (int)out[0].type, nbInput, nbOutput);
    }

    //PUGIN CONSISTENCY CHECKING
public:
    const char *getPluginType() const noexcept override { return "Res2ChannelPad"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

    const char *getPluginNamespace() const noexcept override { return mNamespace.data(); }

//PLUGIN SERIALIZATION
public:

    size_t getSerializationSize() const noexcept override {
        size_t serializationSize = 0;
        serializationSize += sizeof(uint32_t);    //_resolution
        serializationSize += sizeof(uint32_t);    //_rowPadding
        serializationSize += sizeof(uint32_t);    //_columnPadding
        serializationSize += sizeof(uint32_t);    //_inChannels
        serializationSize += sizeof(uint32_t);    //_inHeight
        serializationSize += sizeof(uint32_t);    //_inWidth
        serializationSize += sizeof(static_cast<int>(_mInDataType));

        return serializationSize;
    }

    void serialize(void *buffer) const noexcept override {
        char *d = static_cast<char *>(buffer);
        const char *const a = d;
        write(d, _resolution);
        write(d, _rowPadding);
        write(d, _columnPadding);
        write(d, _inChannels);
        write(d, _inHeight);
        write(d, _inWidth);
        write(d, static_cast<int>(_mInDataType));

        if (d != a + getSerializationSize())
            DPRINTF(1, "Res2ChannelPad serialize error!\n");

    }

//PLUGIN EXECUTION
public:
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

    void terminate() noexcept override {}

    int initialize() noexcept override { return 0; }

    int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace, cudaStream_t stream) noexcept override {
        std::vector<uint32_t> inDims(4, 0);
        inDims[0] = _inChannels;
        inDims[1] = _inHeight;
        inDims[2] = _inWidth;
        inDims[3] = _resolution;
        return Res2ChannelPadForward(batchSize, inputs, outputs, workspace, inDims, _rowPadding, _columnPadding, _mInDataType, stream);

    }

    void destroy() noexcept override { delete this; }

    IPluginV2Ext *clone() const noexcept override {
        auto *plugin = new Res2ChannelPad(*this);
        return plugin;
    }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }

private:
    uint32_t _resolution;
    uint32_t _rowPadding;
    uint32_t _columnPadding;
    uint32_t _inChannels;
    uint32_t _inHeight;
    uint32_t _inWidth;

    DataType _mInDataType;

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

class Res2ChannelPadCreator : public IPluginCreator {
public:
    const char *getPluginName() const noexcept override { return "Res2ChannelPad"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    const PluginFieldCollection *getFieldNames() noexcept override { return &mFieldCollection; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
        auto plugin = new Res2ChannelPad(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
        auto plugin = new Res2ChannelPad(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(Res2ChannelPadCreator);

#endif