#ifndef TRT_MS_DEFORMABLE_ATTENTION_H
#define TRT_MS_DEFORMABLE_ATTENTION_H
#include <cublas_v2.h>
#include <memory>
#include <string>
#include <vector>

// Multi Scale Deformable Attention
class MSDeformableAttention : public nvinfer1::IPluginV2DynamicExt {
 public:
  MSDeformableAttention(const std::string &name, 
        const int* spatial_shapes, int spatial_shapes_length, int mode,
        int paddingMode, bool alignCorners);

  MSDeformableAttention(const std::string name, const void *data, size_t length);

  MSDeformableAttention() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 private:
  const std::string mLayerName;
  std::string mNamespace;
  int mSpatialShapesLength;
  std::vector<int> mSpatialShapes;
  std::vector<int> mLevelStartIndex; 
  int* mSpatialShapes_d;
  int* mLevelStartIndex_d;
  int mMode;
  int mPaddingMode;
  int mAlignCorners;

};

class MSDeformableAttentionCreator : public nvinfer1::IPluginCreator {
 public:
  MSDeformableAttentionCreator();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(
      const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
REGISTER_TENSORRT_PLUGIN(MSDeformableAttentionCreator);
#endif  // TRT_GRID_SAMPLER_HPP
