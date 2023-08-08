/*
 * Copyright (c) 2018, Xmotors.ai. All rights reserved.
 */

DEFINE_BUILTIN_OP_IMPORTER(ATen) {
  OnnxAttrs attrs(node);
  std::string node_name = node.name();
  if (node_name.empty()) {
    node_name = attrs.get<std::string>("operator");
  }

  cout << "Import ATen Name=" << node_name << endl;

  std::vector<nvinfer1::ITensor *> tensors;
  for (auto &input : inputs) {
    if (!input.is_tensor())
      continue;  // ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
#if NV_TENSORRT_MAJOR >= 4
    ASSERT(input.tensor().getType() != nvinfer1::DataType::kINT32,
           ErrorCode::kUNSUPPORTED_NODE);
#endif  // NV_TENSORRT_MAJOR >= 4
    tensors.push_back(&input.tensor());
    cout << "Input Tensor: " << input.tensor().getName() << endl;
  }
  int noutput = node.output().size();
  cout << "Output Num=" << noutput << endl;

  char headchar = node_name.front();

  static nvinfer1::ITensor *im_info;
  if (headchar == 'G')  // GenerateProposalsOp
  {
    if (inputs.size() >= 3) im_info = tensors[2];  // im_info in caffe2
    // just forward
    return {{tensors[0], tensors[1]}};
  } else if (headchar == 'C')  // CollectAndDistributeFpnRpnProposalsOp
  {
    CollectAndDisOpPlugin *pPlugin = nullptr;
    std::vector<nvinfer1::ITensor *> plugin_tensors;  // input for plugin
    int featUsed = 5;
    if (attrs.count("anchor_sizes")) {
      auto an_sizes = attrs.get<std::vector<int>>("anchor_sizes");
      auto an_stride = attrs.get<std::vector<int>>("anchor_stride");
      auto an_ratios = attrs.get<std::vector<float>>("aspect_ratios");
      featUsed = an_stride.size();  // 5 or 4
      char *val = getenv("TRT_FEAT");
      if (NULL != val && 5 == featUsed) {
        featUsed = atoi(val);
        DPRINTF(2, "getenv TRT_FEAT = %d\n", featUsed);
        if (4 == featUsed) {  // only support pruned to 4fpn feat
          an_stride.erase(an_stride.begin());
          an_sizes.erase(an_sizes.begin());
        }  // remove input feat of fpn_res2_2_sum
      }
      cout << "featUsed=" << featUsed << endl;
      pPlugin = new CollectAndDisOpPlugin(an_sizes, an_stride, an_ratios);
    } else {
      pPlugin = new CollectAndDisOpPlugin();
    }
    std::vector<nvinfer1::ITensor *> cls_tensors, bbox_tensors;
    for (int i = 5 - featUsed; i < 5; i++) {
      cls_tensors.push_back(flatten_tensor(ctx, *tensors[i]));
      bbox_tensors.push_back(flatten_tensor(ctx, *tensors[i + 5]));
    }
    auto *layercat1 =
        ctx->network()->addConcatenation(&cls_tensors[0], featUsed);
    auto *layercat2 =
        ctx->network()->addConcatenation(&bbox_tensors[0], featUsed);
    plugin_tensors.push_back(layercat1->getOutput(0));
    plugin_tensors.push_back(layercat2->getOutput(0));
    if (attrs.count("pooled_h")) {  // comined RoIAlign
      int pooled_h = attrs.get<int>("pooled_h");
      int pooled_w = attrs.get<int>("pooled_w", pooled_h);
      int sampling_ratio = attrs.get<int>("sampling_ratio");
      pPlugin->SetRoIAlign(pooled_h, pooled_w, sampling_ratio);
      for (unsigned int i = 10; i < tensors.size(); i++)
        plugin_tensors.push_back(tensors[i]);
    }

    nvinfer1::IPluginLayer *layer = ctx->addPlugin(pPlugin, plugin_tensors);

    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < noutput; ++i) {
      outputs.push_back(layer->getOutput(i));
    }
    return outputs;

  } else if (headchar == 'R')  // RoIAlign
  {
    float spatial_scale = attrs.get<float>("spatial_scale");
    int pooled_h = attrs.get<int>("pooled_h");
    int pooled_w = attrs.get<int>("pooled_w");
    int sampling_ratio = attrs.get<int>("sampling_ratio");
    DPRINTF(1,
            "RoIAlign attrs spatial_scale=%f pooled_h=%d pooled_w=%d "
            "sampling_ratio=%d\n",
            spatial_scale, pooled_h, pooled_w, sampling_ratio);
    RETURN_FIRST_OUTPUT(ctx->addPlugin(
        new RoIAlignPlugin(spatial_scale, pooled_h, pooled_w, sampling_ratio),
        tensors));
    // return {{layerR->getOutput(0)}};
  } else if (headchar == 'B')  // BatchPermutation
  {
    if (inputs.size() >= 3) im_info = tensors[2];
    if (im_info != nullptr) {
      RETURN_FIRST_OUTPUT(ctx->addPlugin(new BatchPermutationPlugin(),
                                         {tensors[0], tensors[1], im_info}));
    } else {
      RETURN_FIRST_OUTPUT(ctx->addPlugin(new BatchPermutationPlugin(),
                                         {tensors[0], tensors[1]}));
    }
  } else if (headchar == 'D')  // DecodeAndNMS
  {
    float nms_thresh = attrs.get<float>("nms_thresh");
    int detections_per_im = attrs.get<int>("detections_per_im");
    auto scales = attrs.get<std::vector<int>>("scales");
    DPRINTF(1, "DecodeAndNMS attrs nms_thresh=%f detections_per_im=%d\n",
            nms_thresh, detections_per_im);
    auto layer = ctx->addPlugin(
        new DecodeAndNMSPlugin(detections_per_im, scales, nms_thresh), tensors);
    std::vector<TensorOrWeights> outputs;
    for (int i = 0; i < noutput; ++i) {
      outputs.push_back(layer->getOutput(i));
    }
    return outputs;
  }

  return {{tensors[0]}};
}

// Convert an ONNX axis into a TRT axis
inline Status convert_axis(int &axis, int nbDims) {
  // Support negative indexing
  if (axis < 0) {
    axis += nbDims;
  }
  // If axis was positive, subtract 1 to strip batch dimension
  else if (axis > 0) {
    axis = axis - 1;
  }
  DPRINTF(1, "convert_axis %d\n", axis);
  ASSERT(axis >= 0 && axis < nbDims, ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

// Returns the input if it is already a tensor. If it is of type ShapedWeights,
// adds a new constant layer to the TRT network and returns its output.
inline nvinfer1::ITensor &convertToTensor(TensorOrWeights &input,
                                          IImporterContext *ctx) {
  if (input.is_tensor()) {
    return input.tensor();
  } else {
    // Handle non-tensor indices input by adding a new constant layer to the
    // network.
    const ShapedWeights &weights = input.weights();
    return *(ctx->network()->addConstant(weights.shape, weights)->getOutput(0));
  }
}

// Takes idx from [MIN_INT, MAX_INT] to [0, ax_size] (for Slice op)
int slice_clip_index(int ax_size, int idx) {
  if (idx < 0) {
    idx += (ax_size + 1);
  }
  return std::min(std::max(idx, 0), ax_size);
}

#if 0  // NV_TENSORRT_MAJOR >= 6 //From onnx_tensorrt Branch 6.0, has some bug
DEFINE_BUILTIN_OP_IMPORTER(Slice) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
  // If opset version >= 10 slice paramerters are weights instead of attributes
  nvinfer1::ITensor& tensor = inputs.at(0).tensor();
  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> axes;
  std::vector<int64_t> steps;
  if(ctx->getOpsetVersion() >= 10)
  {
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    int64_t * array_start = static_cast<int64_t *>(inputs.at(1).weights().values);
    starts = std::vector<int64_t> (array_start, array_start + inputs.at(1).weights().count());
    ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    array_start = static_cast<int64_t *>(inputs.at(2).weights().values);
    ends = std::vector<int64_t> (array_start, array_start + inputs.at(2).weights().count());
    ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    array_start = static_cast<int64_t *>(inputs.at(3).weights().values);
    axes = std::vector<int64_t> (array_start, array_start + inputs.at(3).weights().count());
    ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    array_start = static_cast<int64_t *>(inputs.at(4).weights().values);
    steps = std::vector<int64_t> (array_start, array_start + inputs.at(4).weights().count());
  }
  else
  {
    OnnxAttrs attrs(node);
    starts = attrs.get<std::vector<int64_t>>("starts");
    ends = attrs.get<std::vector<int64_t>>("ends");
    axes = attrs.get<std::vector<int64_t>>("axes");
    steps = std::vector<int64_t>(starts.size(), 1);
  }
  ASSERT(axes.size() == starts.size() && axes.size() == ends.size() && axes.size() == steps.size(), ErrorCode::kINVALID_VALUE);

  const nvinfer1::Dims dims = tensor.getDimensions();
  const int nbDims = dims.nbDims;
  auto makeDims = [nbDims](int initVal)->nvinfer1::Dims{
    nvinfer1::Dims result{nbDims, {},{}};
    std::fill_n(&result.d[0], nbDims, initVal);
    return result;
  };
  nvinfer1::Dims sliceStart = makeDims(0);
  nvinfer1::Dims sliceEnd = dims;
  nvinfer1::Dims sliceSize = dims;
  nvinfer1::Dims sliceStride = makeDims(1); // ONNX has support for strides before opset 10
  for (size_t i = 0; i < axes.size(); i++){
    DPRINTF(2, "Slice %d axes=%ld starts=%ld ends=%ld size=%ld\n", i, axes[i], starts[i], ends[i], steps[i]);
    int axis = axes[i];
    // Convert the axis if it passes the no-op check, we catch actual slices across batch dimension here
    TRT_CHECK(convert_axis(axis, nbDims));

    // Special pass through for no-ops (slice across the whole dimension, [:])
    if (starts[i] == 0 && ends[i] >= dims.d[axis] && steps[i] == 1)
    {
      continue;
    }

    // Check if slice is valid
    ASSERT(steps[i] != 0, ErrorCode::kINVALID_VALUE);
    sliceStride.d[axis] = steps[i];

    // Calculate start index
    // Support for negative indexing
    sliceStart.d[axis] = slice_clip_index(dims.d[axis], starts[i]);
    sliceEnd.d[axis] = slice_clip_index(dims.d[axis], ends[i]);

    sliceSize.d[axis] = std::max(static_cast<int>(std::ceil(static_cast<float>(sliceEnd.d[axis] - sliceStart.d[axis]) / steps[i])), 0);
    DPRINTF(2, "Slice dims.d[i]=%d starts[i]=%d ends[i]=%d size=%d\n", dims.d[axis], sliceStart.d[axis], sliceEnd.d[axis], sliceSize.d[axis]);
  }
  // If entire slice op was a no-op, simply return the input tensor
  if (sliceSize == makeDims(0) || sliceSize == dims)
  {
    DPRINTF(2, "Skip Slice\n");
    return {{&tensor}};
  }
  else
  {
    // Slice layer can't handle size of 0
    for (size_t i = 0; i < axes.size(); i++)
    {
        ASSERT(sliceSize.d[i] != 0, ErrorCode::kINVALID_VALUE);
    }
  }
  RETURN_FIRST_OUTPUT(ctx->network()->addSlice(tensor, sliceStart, sliceSize, sliceStride));
}
#else
DEFINE_BUILTIN_OP_IMPORTER(Slice) {
  ASSERT(inputs.size() == 1, ErrorCode::kINVALID_NODE);
  nvinfer1::ITensor *tensor_ptr = &convertToTensor(inputs.at(0), ctx);
  const nvinfer1::Dims dims = tensor_ptr->getDimensions();
  const int nbDims = dims.nbDims;
  OnnxAttrs attrs(node);
  // We don't support implicit indexing due to
  // inability to deal with batch dim slicing
  // (TRT doesn't support batch dim slicing)
  ASSERT(attrs.count("axes"), ErrorCode::kUNSUPPORTED_NODE);
  auto axes = attrs.get<std::vector<int>>("axes");
  auto starts = attrs.get<std::vector<int>>("starts");
  auto ends = attrs.get<std::vector<int>>("ends");

  // Argument validation
  // Since indexing is explicit, there must be
  // equal number of axis, start and end indices
  ASSERT((nbDims >= 3) && (axes.size() == starts.size()) &&
             (starts.size() == ends.size()),
         ErrorCode::kUNSUPPORTED_NODE);

  auto makeDims = [nbDims](int initVal) -> nvinfer1::Dims {
    nvinfer1::Dims result{nbDims, {}, {}};
    std::fill_n(&result.d[0], nbDims, initVal);
    return result;
  };
  nvinfer1::Dims sliceStart = makeDims(0);
  nvinfer1::Dims sliceEnd = dims;
  nvinfer1::Dims sliceSize = dims;
  nvinfer1::Dims sliceStride = makeDims(1);  // TODO: ONNX opset 10 has Stride
  bool isFullDims = false;                   // Not only H & W
  for (size_t i = 0; i < axes.size(); ++i) {
    DPRINTF(2, "Slice %zu a=%d s=%d e=%d\n", i, axes[i], starts[i], ends[i]);
    int axis = axes[i];
    if (0 == axis) continue;  // skip the batch

    // We don't allow slicing batch dim, due to TRT limitations
    TRT_CHECK(convert_axis(axis, nbDims));

    sliceStart.d[axis] = slice_clip_index(dims.d[axis], starts[i]);
    sliceEnd.d[axis] = slice_clip_index(dims.d[axis], ends[i]);
    sliceSize.d[axis] = sliceEnd.d[axis] - sliceStart.d[axis];
    DPRINTF(2, "Slice clip s=%d e=%d\n", sliceStart.d[axis], sliceEnd.d[axis]);

    // Special pass through for no-ops (slice across the whole dimension, [:])
    if (sliceStart.d[axis] == 0 && sliceEnd.d[axis] >= dims.d[axis]) {
      continue;
    }

    // TRT only supports slicing HW dims when using padding layer,
    // so if user wants to slice some other axis, we check whether
    // slice contains full dimension
    if (axis != nbDims - 2 && axis != nbDims - 1) {
      isFullDims = true;
    }
  }
#if NV_TENSORRT_MAJOR >= 6
  if (isFullDims) {  // Full dims Slice Support by TensorRT6
    RETURN_FIRST_OUTPUT(ctx->network()->addSlice(*tensor_ptr, sliceStart,
                                                 sliceSize, sliceStride));
  } else
#endif
  {
    nvinfer1::DimsHW start_pad, end_pad;
    start_pad.h() = sliceStart.d[nbDims - 2];
    end_pad.h() = dims.d[nbDims - 2] - sliceEnd.d[nbDims - 2];
    start_pad.w() = sliceStart.d[nbDims - 1];
    end_pad.w() = dims.d[nbDims - 1] - sliceEnd.d[nbDims - 1];
    if (start_pad.h() > 0 || start_pad.w() > 0 || end_pad.h() || end_pad.w()) {
      DPRINTF(2, "Slice start(%d,%d) -> end(%d,%d)\n", start_pad.h(),
              start_pad.w(), end_pad.h(), end_pad.w());
      auto layer_ptr =
          ctx->network()->addPadding(*tensor_ptr, -start_pad, -end_pad);
      ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);
      tensor_ptr = layer_ptr->getOutput(0);
    } else
      DPRINTF(2, "Skip Slice\n");
  }
  return {{tensor_ptr}};
}
#endif

#if NV_TENSORRT_MAJOR >= 4
DEFINE_BUILTIN_OP_IMPORTER(Gather) {
  nvinfer1::ITensor &data = convertToTensor(inputs.at(0), ctx);
  nvinfer1::ITensor &indices = convertToTensor(inputs.at(1), ctx);
  OnnxAttrs attrs(node);
  int axis = attrs.get<int>("axis", 0);
  int nbDims = inputs.at(0).shape().nbDims;
  TRT_CHECK(convert_axis(axis, nbDims));
  RETURN_FIRST_OUTPUT(ctx->network()->addGather(data, indices, axis));
}
#endif  // NV_TENSORRT_MAJOR >= 4

ITensor *addGEMVInt8(IImporterContext *ctx, ITensor *input, int nbOutputs,
                     Weights kernelWeights, Weights biasWeights) {
  int nGEMV = 0;
  char *val = getenv("TRT_GEMV");
  if (NULL != val) {
    nGEMV = atoi(val);
    DPRINTF(2, "getenv TRT_GEMV = %d\n", nGEMV);
  }
  if (nGEMV <= 0 || kernelWeights.count < 128) {
    return nullptr;
  }

  nvinfer1::ILayer *layer_ptr = ctx->addPlugin(
      new GemvInt8Plugin(nbOutputs, kernelWeights, biasWeights), {input});
  // ASSERT(layer_ptr, ErrorCode::kUNSUPPORTED_NODE);
  return layer_ptr->getOutput(0);
}

#if NV_TENSORRT_MAJOR >= 6
DEFINE_BUILTIN_OP_IMPORTER(Resize) {
  nvinfer1::ITensor &input = convertToTensor(inputs.at(0), ctx);
  int input_dims = input.getDimensions().nbDims;
  ASSERT(input_dims > 0, ErrorCode::kUNSUPPORTED_NODE);

  // Add resize layer
  nvinfer1::IResizeLayer *layer = ctx->network()->addResize(input);

  // Retrive and validate scale factors.
  // Scale factors include batch dimensions as well.
  ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
  auto scales = inputs.at(1);
  // Support for scales as weights
  ASSERT(scales.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
  ShapedWeights scales_weights = scales.weights();
  ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(scales_weights.count() == static_cast<size_t>(input_dims),
         ErrorCode::kUNSUPPORTED_NODE);
  ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT,
         ErrorCode::kINVALID_NODE);
  // Get floating point scale factors.
  float const *scales_ptr = static_cast<float const *>(scales_weights.values);
  layer->setScales(scales_ptr, input_dims);

  // Set resize mode
  OnnxAttrs attrs(node);
  auto mode = attrs.get<std::string>("mode", "nearest");
  ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
  // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8)
  // resize.
  nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
  if (mode == "linear") {
    // Linear resize support 1-D, 2-D and 3D resize.
    ASSERT((input_dims >= 1) && (input_dims <= 3),
           ErrorCode::kUNSUPPORTED_NODE);
    resizeMode = nvinfer1::ResizeMode::kLINEAR;
  }
  layer->setResizeMode(resizeMode);

  // Set other attributes. ONNX spec does not have this attribute yet.
  // Default: False. Set it any way.
  layer->setAlignCorners(false);

  // Return layer output
  RETURN_FIRST_OUTPUT(layer);
}

#include "UpsampleInt8.h"

DEFINE_BUILTIN_OP_IMPORTER(UpsampleV2) {
  nvinfer1::ITensor &input = convertToTensor(inputs.at(0), ctx);
  int input_dims = input.getDimensions().nbDims;
  ASSERT(input_dims > 2, ErrorCode::kUNSUPPORTED_NODE);
  std::vector<nvinfer1::ITensor *> tensors{&input};

  float height_scale, width_scale;
  OnnxAttrs attrs(node);
  if (!attrs.count("scales")) {
    height_scale = attrs.get<float>("height_scale");
    width_scale = attrs.get<float>("width_scale");
  } else {
    auto scales = attrs.get<std::vector<float>>("scales");
    ASSERT(scales.size() == 4, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(scales[0] == 1 && scales[1] == 1, ErrorCode::kUNSUPPORTED_NODE);
    height_scale = scales[2];
    width_scale = scales[3];
  }

  // Add plugin layer
  IPluginV2 *plugin = new UpsamplePluginV2(width_scale, height_scale);
  auto *layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);

  // Return layer output
  RETURN_FIRST_OUTPUT(layer);
}

// Batch Pad Concatenate op for main/narrow combined onnx
DEFINE_BUILTIN_OP_IMPORTER(BatchConcatPad) {
  std::vector<nvinfer1::ITensor *> tensors;
  std::vector<nvinfer1::Weights> weights;
  nvinfer1::Dims weightshape;
  for (auto &input : inputs) {
    if (input.is_weights()) {  // Add constant layer for weight
      auto weight = input.weights();
      weightshape = remove_dim(weight.shape, BATCH_DIM);
      weights.push_back(weight);
      continue;
    }
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);

    tensors.push_back(&input.tensor());
  }

  int batch = weights.size() / 3;  // 1 batch has 3 weight
  // Add plugin layer
  IPluginV2 *plugin = new BatchPadConcatPlugin(batch, weights, weightshape);
  auto *layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);

  // Return layer output
  RETURN_FIRST_OUTPUT(layer);
}

#endif  // NV_TENSORRT_MAJOR >= 6
