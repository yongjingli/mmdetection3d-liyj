#ifndef CHANNEL2SPATIAL_H_
#define CHANNEL2SPATIAL_H_

#include <math.h>
#include <vector>

#include "common.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <NvInferRuntimeCommon.h>

using namespace nvinfer1;

using namespace std;

extern "C" int Channel2SpatialForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                            int scale, int mixed, std::vector<int> inputDims, DataType dType, TensorFormat format, cudaStream_t stream);
// mixed batch for side: side_front_left x side_rear_left, side_front_right x side_rear_right, 
extern "C" int MixedBatchForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                      int scale, int mixed, std::vector<int> inputDims, DataType dType, 
                      TensorFormat dFormat, cudaStream_t stream);

class Channel2Spatial : public IPluginV2IOExt {
//Constructor
public:
    Channel2Spatial(const PluginFieldCollection *fc) { 
        const PluginField* fields = fc->fields;
        for (auto i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "scale")) {
                assert(fields[i].type == PluginFieldType::kINT32);
                _scale = *static_cast<const int*>(fields[i].data);
                DPRINTF(1, "C2S scale is=%d\n", _scale);
            }
        }
    }

    //This consturctor is used for desereialezed
    Channel2Spatial(void const *data, size_t length) {
        deserialize_value(&data, &length, &_inputDims);    
        deserialize_value(&data, &length, &_scale);
        deserialize_value(&data, &length, &_mixed_batch);
        deserialize_value(&data, &length, (int*)&_mDataType);
        deserialize_value(&data, &length, (int*)&_mTensorFormat);

        if (0 != length) {
          DPRINTF(1, "Channel2Spatial init error! Left length = %d\n", length);
        }
    }

    //This constructor is used for BUILDIN_OP_IMPORTER
    Channel2Spatial(int scale, int mixed_batch):_scale(scale), _mixed_batch(mixed_batch){}

//PLUGIN CONFIGURATION
public:
    int getNbOutputs() const noexcept override { return 1; }

    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const noexcept override {
        assert(inputTypes && nbInputs == 1);
        (void)index;
        return inputTypes[0];
    }

    Dims getOutputDimensions(int index, const Dims *inputDims, int nbInputDims) noexcept override {
        _inputDims.resize(inputDims->nbDims);
        for(int i = 0; i < _inputDims.size(); ++i) {
            _inputDims[i] = inputDims->d[i];
        }
        
        nvinfer1::Dims3 output;
        if ( _mixed_batch > 0 ) {
          output = Dims3(inputDims->d[2] * _scale, inputDims->d[1], inputDims->d[0] / _scale); // W*2, H, C/2
        } else {
          output = Dims3(inputDims->d[0] / _scale, inputDims->d[1] * sqrt(_scale), inputDims->d[2] * sqrt(_scale));
        }
        
        DPRINTF(1, "C2S Output shape=%d(%d,%d,%d)\n", output.nbDims, output.d[0], output.d[1], output.d[2]);
        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const noexcept override {
        assert(nbInputs  >= 1);
        assert(nbOutputs == 1);
        assert(pos < nbInputs + nbOutputs);
        DPRINTF(1, "C2S supportsFormatCombination pos: %d format: %d type: %d\n", pos, inOut[pos].format, inOut[pos].type);
        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        condition     &= inOut[pos].type   != DataType::kINT32;
        // condition     &= inOut[pos].type   != DataType::kINT8;
        condition     &= inOut[pos].type   == inOut[0].type;
        // condition     &= inOut[pos].format == inOut[0].format;
        bool conditionCHW32 = inOut[pos].format == nvinfer1::TensorFormat::kCHW32 ;
        conditionCHW32 &= inOut[pos].type   == nvinfer1::DataType::kINT8;
        conditionCHW32 &= inOut[pos].type   == inOut[0].type;
        // conditionCHW32 &= inOut[pos].format == inOut[0].format;
        return condition || conditionCHW32;
    }

    // using IPluginV2IOExt::configurePlugin;
    void configurePlugin(const PluginTensorDesc *in,  int nbInput,
                         const PluginTensorDesc *out, int nbOutput) noexcept override {
        assert(in  && nbInput  >= 1);
        assert(out && nbOutput == 1);
        for(int i = 0; i < nbInput; ++i) {
            assert(in[i].type   == out[0].type);
            assert(in[i].format == out[0].format);
        }
        assert(out[0].format == TensorFormat::kLINEAR);
        assert(out[0].type   != DataType::kINT32);

        _mDataType = in[0].type;
        
        DPRINTF(1, "C2S configType: In=%d Out=%d num: In=%d Out=%d \n", (int)in[0].type, (int)out[0].type, nbInput, nbOutput);
    }

//PLUGIN SERIALIZATION
public:

    size_t getSerializationSize() const noexcept override {
        size_t size = serialized_size(_inputDims) + serialized_size(_scale) + serialized_size(_mixed_batch);
        size += serialized_size((int)_mDataType) + serialized_size((int)_mTensorFormat);

        return size;
    }

    void serialize(void *buffer) const noexcept override {
      void *d = buffer;
      const char *const a = static_cast<char *>(d);
      
      serialize_value(&d, _inputDims);
      serialize_value(&d, _scale);
      serialize_value(&d, _mixed_batch);
      serialize_value(&d, (int)_mDataType);
      serialize_value(&d, (int)_mTensorFormat);
                
      if (d != a + getSerializationSize()) {
        DPRINTF(1, "Channel2Spatial serialize error!\n");
      }
    }    

//PUGIN CONSISTENCY CHECKING
public:
    const char *getPluginType() const noexcept override { return "Channel2Spatial"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(const char *libNamespace) noexcept override { _mNamespace = libNamespace; }

    const char *getPluginNamespace() const noexcept override { return _mNamespace.data(); }

//PLUGIN EXECUTION
public:
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

    void terminate() noexcept override {}

    int initialize() noexcept override { return 0;}

    int enqueue(int batchSize, const void *const *inputs, void * const *outputs, void *workspace, cudaStream_t stream) noexcept override {
        if (TRT_DEBUGLEVEL == -1) {
            printf("Skip Channel2Spatial::enqueue!!\n");
            return 0;
        }
        
        return Channel2SpatialForward(batchSize, inputs, outputs, workspace, _scale, 1, _inputDims, _mDataType, _mTensorFormat, stream);
    }

    void destroy() noexcept override { delete this; }

    IPluginV2Ext *clone() const noexcept override {
        auto *plugin = new Channel2Spatial(*this);
        return plugin;
    }

public:
    
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override { return false; }

private:
    int              _scale;
    int              _mixed_batch;    
    std::vector<int> _inputDims;
        
    DataType         _mDataType;
    TensorFormat     _mTensorFormat;
    std::string      _mNamespace;
};

class Channel2SpatialCreator : public IPluginCreator {
public:
    Channel2SpatialCreator() {
        mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
    }
    const char *getPluginName() const noexcept override { return "Channel2Spatial"; }

    const char *getPluginVersion() const noexcept override { return "1"; }

    const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override {
        auto plugin = new Channel2Spatial(fc);
        mPluginName = name;
        return plugin;
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override {
        auto plugin = new Channel2Spatial(serialData, serialLength);
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
REGISTER_TENSORRT_PLUGIN(Channel2SpatialCreator);
#endif