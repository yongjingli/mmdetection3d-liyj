/*
 * Copyright (c) 2021, Xpeng Motor. All rights reserved.
 * Upsample Plugin, Int8 version
 */
#ifndef __BATCHCONCATPAD_HPP__
#define __BATCHCONCATPAD_HPP__

#include "common.h"

union my_int256_t
{
  int8_t i8[32];
  longlong4 i256;
};
extern "C" int BatchPadConcatForward(int batchSize, Dims &mInputDims, Dims &mOutputDims, DataType &mInType,
                          DataType &mOutType, void **weights, const void *const *inputs, void *const*outputs,
                          void *ws, cudaStream_t stream);
extern "C" int BatchPadConcatForwardV2(int batchSize, Dims &inDims, Dims outDims, DataType inType, DataType outType,
                            TensorFormat inFormat, TensorFormat outFormat, int cPad, void **weights,
                            const void *const *inputs, void *const*outputs, void *workspace, cudaStream_t stream);
//Batch Pad Concatenate op for main/narrow combined onnx
class BatchPadConcatPlugin : public IPluginV2IOExt
{
public:
  BatchPadConcatPlugin(const PluginFieldCollection *fc) {
    const PluginField* fields = fc->fields;
    std::vector<int> pads{0};
    float value;
    std::vector<float> cam_weight[2][3];
    for (auto i = 0; i < fc->nbFields; ++i)
    {  
      const char* attrName = fields[i].name;
      DPRINTF(2, "BatchPadConcatPlugin attrName=%s length=%d\n", attrName, fields[i].length);
      if (!strcmp(attrName, "pads")) {  
        assert(fields[i].type == PluginFieldType::kINT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          pads.resize(size);
          const auto* aR = static_cast<const int*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              pads[j] = *aR;
              DPRINTF(3, "pads[%d]=%d\n", j, pads[j]);
              aR++;
          }
        }
      } else if (!strcmp(attrName, "value")) {
        assert(fields[i].type == PluginFieldType::kFLOAT32);
        value = *static_cast<const float*>(fields[i].data);
        DPRINTF(3, "value=%f\n", value);
      } else if (!strncmp(attrName, "cam", 3)) {
        int cam_nb = 0; // cam0: narrow, cam1: main -> b0: main, b1: narrow
        if( attrName[3] == '0' ) {
          cam_nb = 1;
        }
        int weight_id = 0; // weight id, 0: cc, 1: f, 2: nc
        if (strstr(attrName, "_f")) {
          weight_id = 1;
        } else if (strstr(attrName, "_nc")) {
          weight_id = 2;
        } 
        DPRINTF(2, "camconv(%s, %d)\n", (cam_nb?"main":"narrow"), weight_id);
        assert(fields[i].type == PluginFieldType::kFLOAT32);
        const int32_t size = fields[i].length;
        if (size > 0) {
          cam_weight[cam_nb][weight_id].resize(size);
          const auto* aR = static_cast<const float*>(fields[i].data);
          for (auto j = 0; j < size; j++) {
              cam_weight[cam_nb][weight_id][j] = *aR;
              aR++;
          }
        }
      } 
    }
      
    mBatch = pads[4];
    mCPad = pads[5];
    mHPad = pads[6];
    mWPad = pads[7];
    DPRINTF(2, "BatchPadConcatPlugin NCHW=(%d,%d,%d,%d)\n", mBatch, mCPad, mHPad, mWPad);
    assert( mBatch <= 2 );
    //copy 3*batch weight from onnx file to CPU
    int wshape[3] = {2,237,457};
    int wnum = 3;
    int height = 240, width = 464;
    mWsize = wnum*wshape[0]*height*width; // 6*240*464
    CPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++)
    {
      CPUWeights[nb].resize(mWsize, 0);
      float *pWout = &CPUWeights[nb][0];
      for (int i = 0; i < wnum; i++)
      {
        const float *pWin = cam_weight[nb][i].data();
        for (int ch = 0; ch < wshape[0]; ch++)
        {
          for (int y = 0; y < wshape[1]; y++)
          {
            memcpy(pWout + y * width, pWin, wshape[2] * sizeof(float));
            pWin += wshape[2];
          }
          pWout += height * width;
        }
      }
    }  
  }

  BatchPadConcatPlugin(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; ++i)
    {
      mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; ++i)
    {
      mOutputDims.d[i] = read<int>(d);
    }

    mInDataType = static_cast<DataType>(read<int>(d));
    mOutDataType = static_cast<DataType>(read<int>(d));
    mBatch = read<int>(d);
    mWsize = read<int>(d);
    CPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++)
    {
      CPUWeights[nb].resize(mWsize);
      float *pW = &CPUWeights[nb][0];
      read<float>(d, pW, mWsize);
    }

    if (mInDataType == DataType::kINT8)
    {
      mInHostScale = read<float>(d);
      mOutHostScale = read<float>(d);
    }
    mCPad = mOutputDims.d[0] - mInputDims.d[0];
    mHPad = mOutputDims.d[1] - mInputDims.d[1];
    mWPad = mOutputDims.d[2] - mInputDims.d[2];
    if (d != a + length){
      DPRINTF(1, "BatchPadConcatPlugin init error!\n");
    }
    DPRINTF(2, "%s InType=%d OutType=%d \n",
            __func__, (int)mInDataType, (int)mOutDataType);
  }

  int mBatch = 0;
  int mWsize = 0;
  std::vector<std::vector<float>> CPUWeights;
  void *GPUWeights[16];
  BatchPadConcatPlugin(int batch, std::vector<Weights> &weights, Dims shape, std::vector<int> padding)
  {
    mBatch = batch;
    mHPad = padding[6];
    mWPad = padding[7];
    int height = shape.d[1] + mHPad;
    int width = shape.d[2] + mWPad;

    //copy 6 weight from onnx file to CPU
    int wnum = weights.size() / mBatch; // 6/2 = 3
    //int wsize = weights[0].count; // 2*237*457
    mCPad = wnum * shape.d[0]; // 3*2 = 6
    //mWsize = wsize * wnum; // 3 * 2*237*457
    mWsize = mCPad * height * width; // 6*240*464
    CPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++)
    {
      CPUWeights[nb].resize(mWsize, 0);
      float *pWout = &CPUWeights[nb][0];
      for (int i = 0; i < wnum; i++)
      {
        const float *pWin = (const float *)weights[nb * wnum + i].values;
        for (int ch = 0; ch < shape.d[0]; ch++)
        {
          for (int y = 0; y < shape.d[1]; y++)
          {
            memcpy(pWout + y * width, pWin, shape.d[2] * sizeof(float));
            pWin += shape.d[2];
          }
          pWout += height * width;
        }
      }
    }
    DPRINTF(2, "%s mBatch=%d mCPad=%d\n", __func__, mBatch, mCPad);
  }

  BatchPadConcatPlugin() {}

  virtual ~BatchPadConcatPlugin() {}

public:
  int getNbOutputs() const noexcept override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs,
                           int nbInputDims) noexcept override
  {
    assert(index == 0 && inputs[0].nbDims == 3);
    int ch = inputs[0].d[0] + mCPad;
    int height = inputs[0].d[1] + mHPad;
    int width = inputs[0].d[2] + mWPad;
    DPRINTF(2, "BatchPadConcatPlugin getOutputDimensions = %d %d %d\n", ch, height, width);
    return Dims3(ch, height, width);
  }

  int initialize() noexcept override
  {
    DPRINTF(2, "%s %d\n", __FUNCTION__, __LINE__);
    // copy weights from CPU to GPU buffer
    void *pBuffer;
    CHECK_CUDA(cudaMalloc(&pBuffer, mWsize * sizeof(float)));
    for (int nb = 0; nb < mBatch; nb++)
    {
      float *pW = nullptr;
      if (mOutDataType == DataType::kHALF)
      {
        CHECK_CUDA(cudaMalloc(&pW, mWsize * sizeof(__half)));
      }
      else
      {
        CHECK_CUDA(cudaMalloc(&pW, mWsize * sizeof(float)));
      }
      cudaStream_t stream = nullptr;
      CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));      
      convertCPU_GPU(&CPUWeights[nb][0], pW, mWsize, (int)mOutDataType, 0, stream, pBuffer);
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);      
      GPUWeights[nb] = pW;
    }
    cudaFree(pBuffer);
    DPRINTF(2, "%s %d\n", __FUNCTION__, __LINE__);
    return 0;
  }

  void terminate() noexcept override
  {
    for (int nb = 0; nb < mBatch; nb++)
    {
      cudaFree(GPUWeights[nb]);
      GPUWeights[nb] = nullptr;
    }
  }

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  int enqueue(int batchSize, void const*const *inputs, void *const*outputs,
              void *workspace, cudaStream_t stream) noexcept override
  {
    return BatchPadConcatForward(batchSize, mInputDims, mOutputDims,
                                 mInDataType, mOutDataType, GPUWeights, inputs, outputs, workspace, stream);
  }

  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(static_cast<int>(mInDataType));
    serializationSize += sizeof(static_cast<int>(mOutDataType));
    serializationSize += sizeof(mBatch);
    serializationSize += sizeof(mWsize);
    serializationSize += mBatch * mWsize * sizeof(float);

    if (mInDataType == DataType::kINT8)
    {
      serializationSize += sizeof(float) * 2;
    }
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

    write(d, static_cast<int>(mInDataType));
    write(d, static_cast<int>(mOutDataType));
    write(d, static_cast<int>(mBatch));
    write(d, static_cast<int>(mWsize));
    for (int nb = 0; nb < mBatch; nb++)
    {
      write(d, &CPUWeights[nb][0], mWsize);
    }

    if (mInDataType == DataType::kINT8)
    {
      write(d, mInHostScale);
      write(d, mOutHostScale);
    }
    if (d != a + getSerializationSize())
      DPRINTF(1, "BatchPadConcatPlugin serialize error!\n");
  }

  void configurePlugin(const PluginTensorDesc *in, int nbInput,
                       const PluginTensorDesc *out, int nbOutput) noexcept override
  {
    //assert(in && nbInput == 1);
    assert(out && nbOutput == 1);
    assert(in[0].format == TensorFormat::kLINEAR &&
           out[0].format == TensorFormat::kLINEAR);

    mInDataType = in[0].type;
    mOutDataType = out[0].type;
    mInputDims = in[0].dims;
    mOutputDims = out[0].dims;
    mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
    mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
    DPRINTF(1, "configType: In=%d Out=%d formatType: In=%d Out=%d \n", (int)mInDataType, (int)mOutDataType, (int)in[0].format, (int)out[0].format);
  }

  //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
  // IOType: FLOAT, HALF, INT8->HALF,
  bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) const noexcept override
  {
    assert(nbOutputs == 1 && pos < nbInputs + nbOutputs);
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;
    condition &= inOut[pos].type != DataType::kINT32;
    if (pos == nbInputs) {  // Output
      condition &= inOut[nbInputs].type != DataType::kINT8;
      condition &= (inOut[nbInputs].type == inOut[0].type) || (DataType::kINT8 == inOut[0].type && DataType::kHALF == inOut[nbInputs].type);
    }
    return condition;
  }

  DataType getOutputDataType(int index, const DataType *inputTypes,
                             int nbInputs) const noexcept override
  {
    assert(inputTypes);
    if (DataType::kINT8 == inputTypes[0])
      return DataType::kHALF;
    else
      return inputTypes[0];
  }

  const char *getPluginType() const noexcept override { return "BatchConcatPad"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new BatchPadConcatPlugin(*this);
    return plugin;
  }

  void setPluginNamespace(const char *libNamespace) noexcept override
  {
    mNamespace = libNamespace;
  }

  const char *getPluginNamespace() const noexcept override { return mNamespace.data(); }

  bool isOutputBroadcastAcrossBatch(int outputIndex,
                                    const bool *inputIsBroadcasted,
                                    int nbInputs) const noexcept override
  {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override
  {
    return false;
  }

private:
  DataType mInDataType;
  DataType mOutDataType;
  Dims mInputDims;
  Dims mOutputDims;
  int mCPad, mHPad, mWPad;
  float mInHostScale{-1.0f};
  float mOutHostScale{-1.0f};
  std::string mNamespace;
};
// Batch Pad Concatenate op for main/narrow combined onnx
class BatchPadConcatV2Plugin : public IPluginV2IOExt
{
public:
  BatchPadConcatV2Plugin(const PluginFieldCollection &fc) { 
  (void)fc; 
  
  }

  BatchPadConcatV2Plugin(const void *data, size_t length)
  {
    const char *d = static_cast<const char *>(data);
    const char *const a = d;
    mInputDims.nbDims = read<int>(d);
    for (int i = 0; i < mInputDims.nbDims; ++i)
    {
      mInputDims.d[i] = read<int>(d);
    }
    mOutputDims.nbDims = read<int>(d);
    for (int i = 0; i < mOutputDims.nbDims; ++i)
    {
      mOutputDims.d[i] = read<int>(d);
    }
    mCPad = read<int>(d);
    mHPad = read<int>(d);
    mWPad = read<int>(d);
    mInDataType = static_cast<DataType>(read<int>(d));
    mOutDataType = static_cast<DataType>(read<int>(d));
    mInTensorFormat = static_cast<TensorFormat>(read<int>(d));
    mOutTensorFormat = static_cast<TensorFormat>(read<int>(d));
    mBatch = read<int>(d);
    mWSize = read<int>(d);

    mCPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++)
    {
      mCPUWeights[nb].resize(mWSize);
      float *pW = &mCPUWeights[nb][0];
      read<float>(d, pW, mWSize);
    }
    if (mInDataType == DataType::kINT8)
    {
      mInHostScale = read<float>(d);
      mOutHostScale = read<float>(d);
    }
    if (d != a + length)
      DPRINTF(1, "BatchPadConcatPlugin init error!\n");
  }

  BatchPadConcatV2Plugin(int batch, std::vector<Weights> &weights, Dims shape, std::vector<int> padding)
  {
    mBatch = batch;
    // customed padding values
    if (padding.size() != 8) {
      DPRINTF(1, "Invalid padding size %zu [8 is needed] \n", padding.size());
    }
    mHPad = padding[6];
    mWPad = padding[7];
    int padHeight = shape.d[1] + mHPad;
    int padWidth = shape.d[2] + mWPad;

    // copy camera conv weights from onnx file to CPU
    mCPad = weights.size();                // 6
    int wNum = weights.size() / mBatch;    // 6 / 2 = 3
    mWSize = mCPad * padHeight * padWidth; // 6 * 240 * 464

    mCPUWeights.resize(mBatch);
    for (int nb = 0; nb < mBatch; nb++)
    {
      mCPUWeights[nb].resize(mWSize, 0);
      float *pWOut = &mCPUWeights[nb][0];
      for (int iNum = 0; iNum < wNum; iNum++)
      {
        const float *pWIn = (const float *)weights[nb * wNum + iNum].values;
        for (int ic = 0; ic < shape.d[0]; ic++)
        {
          for (int ih = 0; ih < shape.d[1]; ih++)
          {
            memcpy(pWOut + ih * padWidth, pWIn, shape.d[2] * sizeof(float));
            pWIn += shape.d[2];
          }
          pWOut += padHeight * padWidth;
        }
      }
    }
    DPRINTF(1, "%s mBatch=%d mCPad=%d  mWSize=%d \n", __func__, mBatch, mCPad, mWSize);
  }

  BatchPadConcatV2Plugin() {}

  virtual ~BatchPadConcatV2Plugin() {}

public:
  int getNbOutputs() const noexcept override { return 1; }

  Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) noexcept override
  {
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    int ch = inputs[0].d[0] + mCPad;
    int height = inputs[0].d[1] + mHPad;
    int width = inputs[0].d[2] + mWPad;
    return Dims3(ch, height, width);
  }

  int initialize() noexcept override
  {
    DPRINTF(2, "%s %d\n", __func__, __LINE__);
    // copy weights from CPU to GPU buffer
    void *pBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&pBuffer, mWSize * sizeof(float)));
    for (int nb = 0; nb < mBatch; nb++)
    {
      float *pW = nullptr;
      if (mOutDataType == DataType::kHALF)
      {
        CHECK_CUDA(cudaMalloc(&pW, mWSize * sizeof(__half)));
      }
      else if (mOutDataType == DataType::kFLOAT)
      {
        CHECK_CUDA(cudaMalloc(&pW, mWSize * sizeof(float)));
      }
      else if (mOutDataType == DataType::kINT8)
      {
        CHECK_CUDA(cudaMalloc(&pW, mWSize));
      }
      convertCPU_GPU(&mCPUWeights[nb][0], pW, mWSize, (int)mOutDataType, 0, (cudaStream_t)0x2, pBuffer);
      mGPUWeights[nb] = pW;
    }
    cudaFree(pBuffer);

    // TODO(xuzn): change ochan acoording to TensorFormat must be excute here.
    if (TensorFormat::kCHW4 == mOutTensorFormat)
    {
      mOutputDims.d[0] = (mOutputDims.d[0] + 3) / 4 * 4;
    }
    else if (TensorFormat::kCHW32 == mOutTensorFormat)
    {
      mOutputDims.d[0] = (mOutputDims.d[0] + 31) / 32 * 32;
    }
    DPRINTF(2, "%s %d\n", __func__, __LINE__);
    return 0;
  }

  void terminate() noexcept override
  {
    for (int nb = 0; nb < mBatch; nb++)
    {
      cudaFree(mGPUWeights[nb]);
      mGPUWeights[nb] = nullptr;
    }
  }

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

  int enqueue(int batchSize, const void *const *inputs, void *const*outputs, void *workspace, cudaStream_t stream) noexcept override
  {
    return BatchPadConcatForwardV2(batchSize, mInputDims, mOutputDims, mInDataType, mOutDataType, mInTensorFormat,
                                   mOutTensorFormat, mCPad, mGPUWeights, inputs, outputs, workspace, stream);
  }

  size_t getSerializationSize() const noexcept override
  {
    size_t serializationSize = 0;
    serializationSize += sizeof(mInputDims.nbDims);
    serializationSize += sizeof(mInputDims.d[0]) * mInputDims.nbDims;
    serializationSize += sizeof(mOutputDims.nbDims);
    serializationSize += sizeof(mOutputDims.d[0]) * mOutputDims.nbDims;
    serializationSize += sizeof(mCPad);
    serializationSize += sizeof(mHPad);
    serializationSize += sizeof(mWPad);
    serializationSize += sizeof(static_cast<int>(mInDataType));
    serializationSize += sizeof(static_cast<int>(mOutDataType));
    serializationSize += sizeof(static_cast<int>(mInTensorFormat));
    serializationSize += sizeof(static_cast<int>(mOutTensorFormat));
    serializationSize += sizeof(mBatch);
    serializationSize += sizeof(mWSize);
    serializationSize += mBatch * mWSize * sizeof(float);

    if (mInDataType == DataType::kINT8)
    {
      serializationSize += sizeof(float) * 2;
    }
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
    write(d, mCPad);
    write(d, mHPad);
    write(d, mWPad);
    write(d, static_cast<int>(mInDataType));
    write(d, static_cast<int>(mOutDataType));
    write(d, static_cast<int>(mInTensorFormat));
    write(d, static_cast<int>(mOutTensorFormat));
    write(d, static_cast<int>(mBatch));
    write(d, static_cast<int>(mWSize));
    for (int nb = 0; nb < mBatch; nb++)
    {
      write(d, &mCPUWeights[nb][0], mWSize);
    }

    if (mInDataType == DataType::kINT8)
    {
      write(d, mInHostScale);
      write(d, mOutHostScale);
    }
    if (d != a + getSerializationSize())
      DPRINTF(1, "BatchPadConcatPlugin serialize error!\n");
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

    mInHostScale = in[0].scale >= 0.0f ? in[0].scale : -1.0f;
    mOutHostScale = out[0].scale >= 0.0f ? out[0].scale : -1.0f;
    DPRINTF(2, "%s Datatype: In=%d Out=%d, Format In=%d Out=%d \n", __func__, (int)mInDataType, (int)mOutDataType,
            (int)mInTensorFormat, (int)mOutTensorFormat);
    DPRINTF(2, "Input dim: 1=%d, 2=%d, 3=%d; Output dim: 1=%d, 2=%d, 3=%d\n", in[0].dims.d[0], in[0].dims.d[1],
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
      condition &= ((inOut[1].format == TensorFormat::kLINEAR) ||
                    ((inOut[1].format == TensorFormat::kCHW4) && (inOut[1].type == DataType::kINT8)) ||
                    ((inOut[1].format == TensorFormat::kCHW32) && (inOut[1].type == DataType::kINT8)));
      condition &= (inOut[1].type == inOut[0].type);
    }
    DPRINTF(2, "In type: %d, format: %d; Out type: %d, format: %d  Condition: %d\n", (int)inOut[0].type, (int)inOut[0].format,
            (int)inOut[1].type, (int)inOut[1].format, (int)condition);
    DPRINTF(2, "Input dim: 1=%d, 2=%d, 3=%d; Output dim: 1=%d, 2=%d, 3=%d\n", inOut[0].dims.d[1], inOut[0].dims.d[2],
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

  const char *getPluginType() const noexcept override { return "BatchConcatPadV2"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  void destroy() noexcept override { delete this; }

  IPluginV2Ext *clone() const noexcept override
  {
    auto *plugin = new BatchPadConcatV2Plugin(*this);
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
  int mCPad{0}, mHPad{0}, mWPad{0}; // camConv channel padding; input height padding; input width padding
  float mInHostScale{-1.0f};
  float mOutHostScale{-1.0f};
  int mBatch{0};                               // input batch size
  int mWSize{0};                               // total input weights's size
  std::vector<std::vector<float>> mCPUWeights; // CPU memory to save weights
  void *mGPUWeights[16];                       // GPU memory to save weights
  std::string mNamespace;
};
class BatchPadConcatPluginCreator : public IPluginCreator
{
public:
  BatchPadConcatPluginCreator(){
    mPluginAttributes.emplace_back(PluginField("pads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("value", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cam1_cc", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cam1_nc", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cam1_f", nullptr, PluginFieldType::kFLOAT32, 1)); 
    mPluginAttributes.emplace_back(PluginField("cam0_cc", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cam0_nc", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("cam0_f", nullptr, PluginFieldType::kFLOAT32, 1)); 
            
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }
  
  const char *getPluginName() const noexcept override { return "BatchConcatPad"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    auto plugin = new BatchPadConcatPlugin(fc);
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new BatchPadConcatPlugin(serialData, serialLength);
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

class BatchPadConcatV2PluginCreator : public IPluginCreator
{
public:
  const char *getPluginName() const noexcept override { return "BatchConcatPadV2"; }

  const char *getPluginVersion() const noexcept override { return "1"; }

  const PluginFieldCollection *getFieldNames() noexcept override { return &mFieldCollection; }

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    auto plugin = new BatchPadConcatV2Plugin(*fc);
    mFieldCollection = *fc;
    mPluginName = name;
    return plugin;
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    auto plugin = new BatchPadConcatV2Plugin(serialData, serialLength);
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

REGISTER_TENSORRT_PLUGIN(BatchPadConcatPluginCreator);
REGISTER_TENSORRT_PLUGIN(BatchPadConcatV2PluginCreator);

#endif // __BATCHCONCATPAD_HPP__