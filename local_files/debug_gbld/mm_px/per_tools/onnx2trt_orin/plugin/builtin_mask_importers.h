/*
 * Copyright (c) 2018, Xmotors.ai. All rights reserved.
 */

#include "builtin_op_importers.hpp"
#include "onnx2trt.hpp"
#include "BatchConcatPad.h"
#include "UpsampleInt8.h"
#include "ConfidenceFilter.h"
#include "gemvInt8.h"
#include "ConvLSTM.h"
#include "ResizeNearest.h"

// bool transposeWeightsGemm(ShapedWeights const& weights,
//                           nvinfer1::Permutation const& perm,
//                           ShapedWeights* result) {
//   nvinfer1::Dims shape = weights.shape;
//   nvinfer1::Dims new_shape;
//   new_shape.nbDims = shape.nbDims;
//   for( int d=0; d<shape.nbDims; ++d ) {
//     new_shape.d[d] = shape.d[perm.order[d]];
//     result->shape.d[d] = new_shape.d[d];
//   }
//   // TODO: Need to generalize this transpose implementation
//   assert(perm.order[0] == 1 && perm.order[1] == 0);
//   if( shape.nbDims == 2 &&
//       weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT ) {
//     for( int i=0; i<new_shape.d[0]; ++i ) {
//       for( int j=0; j<new_shape.d[1]; ++j ) {
//         float const* src = (float*)weights.values;
//         float*       dst = (float*)result->values;
//         int src_stride = weights.shape.d[1];
//         int dst_stride = result->shape.d[1];
//         dst[i * dst_stride + j] = src[j * src_stride + i];
//       }
//     }
//   } else {
//     // TODO: Implement general transposes and multiple data types
//     // Unsupported weights transpose
//     return false;
//   }
//   return true;
// }

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
  DPRINTF(2, "convert_axis %d\n", axis);
  ASSERT(axis >= 0 && axis < nbDims, ErrorCode::kUNSUPPORTED_NODE);
  return Status::success();
}

// // Returns the input if it is already a tensor. If it is of type ShapedWeights,
// // adds a new constant layer to the TRT network and returns its output.
// inline nvinfer1::ITensor &convertToTensor(TensorOrWeights &input, IImporterContext *ctx) {
//   if (input.is_tensor()) {
//     return input.tensor();
//   } else {
//     // Handle non-tensor indices input by adding a new constant layer to the
//     // network.
//     const ShapedWeights &weights = input.weights();
//     return *(ctx->network()->addConstant(weights.shape, weights)->getOutput(0));
//   }
// }

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
    nvinfer1::Dims result{nbDims,{}};
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
// DEFINE_BUILTIN_OP_IMPORTER(Slice) {
//   ASSERT(inputs.size() == 1, ErrorCode::kINVALID_NODE);
//   nvinfer1::ITensor *tensor_ptr = &convertToTensor(inputs.at(0), ctx);
//   const nvinfer1::Dims dims = tensor_ptr->getDimensions();
//   const int nbDims = dims.nbDims;
//   OnnxAttrs attrs(node, ctx);
//   // We don't support implicit indexing due to
//   // inability to deal with batch dim slicing
//   // (TRT doesn't support batch dim slicing)
//   ASSERT(attrs.count("axes"), ErrorCode::kUNSUPPORTED_NODE);
//   auto axes = attrs.get<std::vector<int>>("axes");
//   auto starts = attrs.get<std::vector<int>>("starts");
//   auto ends = attrs.get<std::vector<int>>("ends");

//   // Argument validation
//   // Since indexing is explicit, there must be
//   // equal number of axis, start and end indices
//   ASSERT((nbDims >= 3) && (axes.size() == starts.size()) && (starts.size() == ends.size()),
//          ErrorCode::kUNSUPPORTED_NODE);

//   auto makeDims = [nbDims](int initVal) -> nvinfer1::Dims {
//     nvinfer1::Dims result{nbDims,{}};
//     std::fill_n(&result.d[0], nbDims, initVal);
//     return result;
//   };
//   nvinfer1::Dims sliceStart = makeDims(0);
//   nvinfer1::Dims sliceEnd = dims;
//   nvinfer1::Dims sliceSize = dims;
//   nvinfer1::Dims sliceStride = makeDims(1);  // TODO: ONNX opset 10 has Stride
//   bool isFullDims = false;                   // Not only H & W
//   for (size_t i = 0; i < axes.size(); ++i) {
//     DPRINTF(2, "Slice %zu a=%d s=%d e=%d\n", i, axes[i], starts[i], ends[i]);
//     int axis = axes[i];
//     if (0 == axis) continue;  // skip the batch

//     // We don't allow slicing batch dim, due to TRT limitations
//     CHECK(convert_axis(axis, nbDims));

//     sliceStart.d[axis] = slice_clip_index(dims.d[axis], starts[i]);
//     sliceEnd.d[axis] = slice_clip_index(dims.d[axis], ends[i]);
//     sliceSize.d[axis] = sliceEnd.d[axis] - sliceStart.d[axis];
//     DPRINTF(2, "Slice clip s=%d e=%d\n", sliceStart.d[axis], sliceEnd.d[axis]);

//     // Special pass through for no-ops (slice across the whole dimension, [:])
//     if (sliceStart.d[axis] == 0 && sliceEnd.d[axis] >= dims.d[axis]) {
//       continue;
//     }

//     // TRT only supports slicing HW dims when using padding layer,
//     // so if user wants to slice some other axis, we check whether
//     // slice contains full dimension
//     if (axis != nbDims - 2 && axis != nbDims - 1) {
//       isFullDims = true;
//     }
//   }
// #if NV_TENSORRT_MAJOR >= 6
//   if (isFullDims) {  // Full dims Slice Support by TensorRT6
//     RETURN_FIRST_OUTPUT(ctx->network()->addSlice(*tensor_ptr, sliceStart, sliceSize, sliceStride));
//   } else
// #endif
//   {
//     nvinfer1::DimsHW start_pad, end_pad;
//     start_pad.h() = sliceStart.d[nbDims - 2];
//     end_pad.h() = dims.d[nbDims - 2] - sliceEnd.d[nbDims - 2];
//     start_pad.w() = sliceStart.d[nbDims - 1];
//     end_pad.w() = dims.d[nbDims - 1] - sliceEnd.d[nbDims - 1];
//     if (start_pad.h() > 0 || start_pad.w() > 0 || end_pad.h() || end_pad.w()) {
//       DPRINTF(2, "Slice start(%d,%d) -> end(%d,%d)\n", start_pad.h(), start_pad.w(), end_pad.h(), end_pad.w());
//       RETURN_FIRST_OUTPUT(ctx->network()->addPadding(*tensor_ptr, -start_pad, -end_pad));
//     } else
//       DPRINTF(2, "Skip Slice\n");
//   }
//   return {{tensor_ptr}};
// }
#endif

#if NV_TENSORRT_MAJOR >= 4
// DEFINE_BUILTIN_OP_IMPORTER(Gather) {
//   nvinfer1::ITensor &data = convertToTensor(inputs.at(0), ctx);
//   nvinfer1::ITensor &indices = convertToTensor(inputs.at(1), ctx);
//   OnnxAttrs attrs(node);
//   int axis = attrs.get<int>("axis", 0);
//   int nbDims = inputs.at(0).shape().nbDims;
//   TRT_CHECK(convert_axis(axis, nbDims));
//   RETURN_FIRST_OUTPUT(ctx->network()->addGather(data, indices, axis));
// }
#endif  // NV_TENSORRT_MAJOR >= 4

// //#if NV_TENSORRT_MAJOR >= 6 // only support tensorrt6.
// DEFINE_BUILTIN_OP_IMPORTER(Resize) {
//   nvinfer1::ITensor &input = convertToTensor(inputs.at(0), ctx);
//   int input_dims = input.getDimensions().nbDims;
//   ASSERT(input_dims > 0, ErrorCode::kUNSUPPORTED_NODE);

//   // Add resize layer
//   nvinfer1::IResizeLayer *layer = ctx->network()->addResize(input);

//   // Retrive and validate scale factors.
//   // Scale factors include batch dimensions as well.
//   ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
//   auto scales = inputs.at(1);
//   // Support for scales as weights
//   ASSERT(scales.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
//   ShapedWeights scales_weights = scales.weights();
//   ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
//   ASSERT(scales_weights.count() == static_cast<size_t>(input_dims), ErrorCode::kUNSUPPORTED_NODE);
//   ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
//   // Get floating point scale factors.
//   float const *scales_ptr = static_cast<float const *>(scales_weights.values);
//   layer->setScales(scales_ptr, input_dims);

//   // Set resize mode
//   OnnxAttrs attrs(node, ctx);
//   auto mode = attrs.get<std::string>("mode", "nearest");
//   ASSERT(mode == "nearest" || mode == "linear", ErrorCode::kUNSUPPORTED_NODE);
//   // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8)
//   // resize.
//   nvinfer1::ResizeMode resizeMode = nvinfer1::ResizeMode::kNEAREST;
//   if (mode == "linear") {
//     // Linear resize support 1-D, 2-D and 3D resize.
//     ASSERT((input_dims >= 1) && (input_dims <= 3), ErrorCode::kUNSUPPORTED_NODE);
//     resizeMode = nvinfer1::ResizeMode::kLINEAR;
//   }
//   layer->setResizeMode(resizeMode);

//   // Set other attributes. ONNX spec does not have this attribute yet.
//   // Default: False. Set it any way.
//   layer->setAlignCorners(false);

//   // Return layer output
//   RETURN_FIRST_OUTPUT(layer);
// }

// Math function op code from onnxtrt6
// Helper for ArgMax/ArgMin
// NodeImportResult argMinMaxHelper(IImporterContext *ctx, ::ONNX_NAMESPACE::NodeProto const& node,
// // NodeImportResult argMinMaxHelper(IImporterContext *ctx, const ::ONNX_NAMESPACE::NodeProto &node,
//                                  std::vector<TensorOrWeights> &inputs, nvinfer1::TopKOperation op) {
//   nvinfer1::ITensor &tensor = convertToTensor(inputs.at(0), ctx);
//   ASSERT(tensor.getType() != nvinfer1::DataType::kINT32, ErrorCode::kUNSUPPORTED_NODE);
//   // Get attributes.
//   OnnxAttrs attrs(node, ctx);
//   int keepdims = attrs.get("keepdims", 1);
//   int axis = attrs.get("axis", 0);

//   int nbDims = tensor.getDimensions().nbDims;
//   // Adjust axis to TensorRT format
//   CHECK(convert_axis(axis, nbDims));

//   uint32_t axisMask = 1 << axis;
//   // Insert a TopK layer with k set to 1.
//   nvinfer1::ITopKLayer *layer = ctx->network()->addTopK(tensor, op, 1, axisMask);
//   ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
//   // Keep the value. Modified: // We don't care about the TopK values, just the indices.
//   nvinfer1::ITensor *value = layer->getOutput(1);  // indices
//   layer->setName(GenLayerName(node, "TopK.argMinMax").c_str());
//   if (keepdims) {
//     // The default behavior of the TopK layer is to keepdims.
//     return {{value}};
//   } else {
//     // Otherwise, we need to squeeze the axis dimension - we achieve this by reshaping.
//     // The new dimensions are just the old dimensions with all values after axis shifted over.
//     nvinfer1::Dims reshapeDims = value->getDimensions();
//     --reshapeDims.nbDims;
//     // The axis dimension should be reduced to size 1 after performing the reduction.
//     ASSERT(reshapeDims.d[axis] == 1, ErrorCode::kINVALID_VALUE);
//     for (int i = axis; i < reshapeDims.nbDims; ++i) {
//       reshapeDims.d[i] = reshapeDims.d[i + 1];
//     }
//     nvinfer1::IShuffleLayer *squeezeLayer = ctx->network()->addShuffle(*value);
//     squeezeLayer->setReshapeDimensions(reshapeDims);
//     return {{squeezeLayer->getOutput(0)}};
//   }
// }

// DEFINE_BUILTIN_OP_IMPORTER(ArgMax) { return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMAX); }

// DEFINE_BUILTIN_OP_IMPORTER(ArgMin) { return argMinMaxHelper(ctx, node, inputs, nvinfer1::TopKOperation::kMIN); }

// DEFINE_BUILTIN_OP_IMPORTER(Asin) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASIN); }

// DEFINE_BUILTIN_OP_IMPORTER(Asinh) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kASINH); }

// DEFINE_BUILTIN_OP_IMPORTER(Atan) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATAN); }

// DEFINE_BUILTIN_OP_IMPORTER(Atanh) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kATANH); }

// DEFINE_BUILTIN_OP_IMPORTER(Cos) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOS); }

// DEFINE_BUILTIN_OP_IMPORTER(Cosh) { return unaryHelper(ctx, node, inputs, nvinfer1::UnaryOperation::kCOSH); }

// Batch Pad Concatenate op for main/narrow combined onnx
DEFINE_BUILTIN_OP_IMPORTER(BatchConcatPad)
{
  std::vector<nvinfer1::ITensor *> tensors;
  std::vector<nvinfer1::Weights> weights;
  nvinfer1::Dims weightshape;
  for (auto &input : inputs)
  {
    if (input.is_weights())
    { // Add constant layer for weight
      auto weight = input.weights();
      DPRINTF(1, "BatchConcatPad weights' dtype %d\n", weight.type);
      weightshape = remove_dim(weight.shape, BATCH_DIM);
      weights.push_back(weight);
      continue;
    }
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    DPRINTF(1, "BatchConcatPad tensor' dtype %d\n", (int)input.tensor().getType());
    tensors.push_back(&input.tensor());
  }

  OnnxAttrs attrs(node, ctx);
  std::vector<int> onnx_padding(8, 0);
  if (attrs.count("pads"))
  {
    onnx_padding = attrs.get<std::vector<int>>("pads");
    ASSERT(onnx_padding.size() == 8, ErrorCode::kUNSUPPORTED_NODE);
  }

  int batch = weights.size() / 3; // 1 batch has 3 weight
  // int version_num = attrs.get<int>("version", 0);
  // // Add plugin layer
  // if (1 == version_num)
  // {
  //   IPluginV2 *plugin = new BatchPadConcatV2Plugin(batch, weights, weightshape, onnx_padding);
  //   auto *layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);
  //   layer->setName(GenLayerName(node, "PluginV2.BatchConcatPadV2").c_str());
  //   return {{layer->getOutput(0)}};
  // }
  IPluginV2 *plugin = new BatchPadConcatPlugin(batch, weights, weightshape, onnx_padding);
  auto *layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);
  layer->setName(GenLayerName(node, "PluginV2.BatchConcatPad").c_str());
  // Return layer output
  return {{layer->getOutput(0)}};
}

DEFINE_BUILTIN_OP_IMPORTER(Upsample)
{
  OnnxAttrs attrs(node, ctx);
  int up_ver = 2;  // 0: Resize, 2:"PluginV2.Upsample"
  up_ver = attrs.get<int>("version", 2);
  char *val = getenv("TRT_UPSAMPLE");  // use upsample plugin
  if (NULL != val) {
    up_ver = atoi(val);
  }

  // use Upsample layer of our plugin
  if (up_ver == 2) {
    ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::ITensor &tensor = inputs.at(0).tensor();
    ASSERT(tensor.getDimensions().nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    auto mode = attrs.get<std::string>("mode", "nearest");

    if (mode == "nearest") {
      float height_scale, width_scale;
      if (ctx->getOpsetVersion() >= 9) {
        ASSERT(inputs.size() == 2, ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        ASSERT(scales_input.is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        ShapedWeights scales_weights = scales_input.weights();
        ASSERT(scales_weights.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(scales_weights.count() == 4, ErrorCode::kUNSUPPORTED_NODE);
        ASSERT(scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT, ErrorCode::kINVALID_NODE);
        float const *scales_ptr = static_cast<float const *>(scales_weights.values);
        ASSERT(scales_ptr[0] == 1 && scales_ptr[1] == 1, ErrorCode::kUNSUPPORTED_NODE);
        height_scale = scales_ptr[2];
        width_scale = scales_ptr[3];
        DPRINTF(1, "height_scale=%f, width_scale=%f\n", height_scale, width_scale);
      } else {
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
      }

      std::vector<nvinfer1::ITensor *> tensors{&tensor};
      // Add plugin v2 layer
      IPluginV2 *plugin = new UpsamplePluginV2(width_scale, height_scale);
      auto layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);
      layer->setName(GenLayerName(node, "PluginV2.Upsample").c_str());
      return {{layer->getOutput(0)}};
    }
  }

  nvinfer1::ITensor& tensor = convertToTensor(inputs.at(0), ctx);
    // TRT does not support BOOL input types for this node
    ASSERT((tensor.getType() != nvinfer1::DataType::kINT32 && tensor.getType() != nvinfer1::DataType::kBOOL)
            && "This version of TensorRT does not support INT32 or BOOL input for the Upsample operator.",
        ErrorCode::kUNSUPPORTED_NODE);
    const int32_t nbDims = tensor.getDimensions().nbDims;
    ASSERT((nbDims > 0) && "The input tensor cannot be a scalar.", ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::IResizeLayer* const layer = ctx->network()->addResize(tensor);
    auto mode = attrs.get<std::string>("mode", "nearest");
    ASSERT((mode == "nearest" || mode == "linear" || mode == "bilinear")
            && "The attribute mode can only be nearest, linear, or bilinear.",
        ErrorCode::kUNSUPPORTED_NODE);
    // Set default resize mode. Nearest resize support N-D (where 0 < N <= 8) resize.
    nvinfer1::ResizeMode resizeMode
        = (mode == "linear" || mode == "bilinear") ? nvinfer1::ResizeMode::kLINEAR : nvinfer1::ResizeMode::kNEAREST;

    if (ctx->getOpsetVersion() >= 9)
    {
        // Get scale factors from inputs[1]
        ASSERT((inputs.size() == 2) && "Operator Upsample requires exactly 2 inputs.", ErrorCode::kINVALID_NODE);
        auto scales_input = inputs.at(1);
        if (scales_input.is_weights())
        {
            // TRT-15340: Remove this and use else path when safety support nbDims == 1.
            ShapedWeights scales_weights = scales_input.weights();
            ASSERT((scales_weights.shape.nbDims == 1) && "The scales input must be 1D.", ErrorCode::kUNSUPPORTED_NODE);
            // Scale factors has batch dimension.
            ASSERT((scales_weights.count() == static_cast<size_t>(nbDims))
                    && "The shape of the scales input must aligin with the dimensions of the input.",
                ErrorCode::kUNSUPPORTED_NODE);
            ASSERT((scales_weights.type == ::ONNX_NAMESPACE::TensorProto::FLOAT)
                    && "This version of TensorRT only supports FLOAT scales input.",
                ErrorCode::kINVALID_NODE);
            float const* scales_ptr = static_cast<float const*>(scales_weights.values);
            std::vector<float> scale_factors(nbDims, 1.0F);
            for (int32_t i = 0; i < nbDims; i++)
            {
                scale_factors[i] = scales_ptr[i];
            }
            if (mode == "linear" || mode == "bilinear")
            {
                ASSERT(canUseLinearResize(scale_factors.size(), &scale_factors.front())
                        && "This version of TensorRT only supports linear resizing on the outermost 3 dimensions",
                    ErrorCode::kUNSUPPORTED_NODE);
            }
            layer->setScales(scale_factors.data(), nbDims);
        }
        else
        {
            nvinfer1::ITensor* resizeShape = resizeShapeTensor(ctx, tensor, scales_input);
            nvinfer1::Dims const outDims = resizeShape->getDimensions();
            ASSERT((outDims.nbDims == 1) && "The scales input must be 1D.", ErrorCode::kUNSUPPORTED_NODE);
            // Scale factors has batch dimension.
            ASSERT((outDims.d[0] == nbDims)
                    && "The shape of the scales input must aligin with the dimensions of the input.",
                ErrorCode::kUNSUPPORTED_NODE);
            ASSERT(
                (resizeShape->getType() == nvinfer1::DataType::kINT32) && "Resize output shape type must be integral.",
                ErrorCode::kINVALID_NODE);
            layer->setInput(1, *resizeShape);
        }
    }
    else
    {
        // TRT-15340: Adapt to use resizeShapeTensor instead when safety support nbDims == 1.
        ASSERT(attrs.count("scales") && "Attribute scales is missing.", ErrorCode::kUNSUPPORTED_NODE);
        // Get scale factors from OnnxAttrs.
        auto scales = attrs.get<std::vector<float>>("scales");
        // Scale factors has batch dimension.
        ASSERT((static_cast<int32_t>(scales.size()) == nbDims)
                && "The shape of the scales input must aligin with the dimensions of the input.",
            ErrorCode::kUNSUPPORTED_NODE);
        std::vector<float> scale_factors(nbDims, 1.0F);
        for (int32_t i = 0; i < nbDims; i++)
        {
            scale_factors[i] = scales[i];
        }
        if (mode == "linear" || mode == "bilinear")
        {
            ASSERT(canUseLinearResize(scale_factors.size(), &scale_factors.front())
                    && "This version of TensorRT only supports linear resizing on the outermost 3 dimensions",
                ErrorCode::kUNSUPPORTED_NODE);
        }
        layer->setScales(scale_factors.data(), nbDims);
    }
    ctx->registerLayer(layer, getNodeName(node));
    layer->setResizeMode(resizeMode);
    layer->setSelectorForSinglePixel(nvinfer1::ResizeSelector::kFORMULA);
    layer->setNearestRounding(nvinfer1::ResizeRoundMode::kFLOOR);
    layer->setCoordinateTransformation(nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    RETURN_FIRST_OUTPUT(layer);
}

// ITensor *addGEMVInt8(IImporterContext *ctx, ::ONNX_NAMESPACE::NodeProto const &node, nvinfer1::ITensor *input, int nbOutputs,
//                      Weights kernelWeights, Weights biasWeights)
// {
//   int nGEMV = 0;
//   char *val = getenv("TRT_GEMV");
//   if (NULL != val)
//   {
//     nGEMV = atoi(val);
//     DPRINTF(2, "getenv TRT_GEMV = %d\n", nGEMV);
//   }
//   auto dims = input->getDimensions();
//   if (nGEMV <= 0 || kernelWeights.count < 128 || (dims.nbDims > 2 && dims.d[2] > 1))
//   {
//     return nullptr;
//   }
//   std::vector<nvinfer1::ITensor *> tensors{input};
//   IPluginV2 *plugin = new GemvInt8PluginV2(nbOutputs, kernelWeights, biasWeights);
//   auto layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);

//   layer->setName(GenLayerName(node, "PluginV2.GemvInt8").c_str());
//   return layer->getOutput(0);
// }
DEFINE_BUILTIN_OP_IMPORTER(Gemm)
{
    // get TRT_GEMV
    // 0: DLA fully connect; 2: GemvInt8 plugin
    int nGEMV = -1;
    char *val = getenv("TRT_GEMV");
    if (NULL != val)
    {
      nGEMV = atoi(val);
      DPRINTF(2, "getenv TRT_GEMV = %d\n", nGEMV);
    }

    // use GemvInt8 plugin
    if (nGEMV == 2) {
      // Note: Currently this only supports A=tensor, B=weights, C=biases
      ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::ITensor *tensor_ptr = &inputs.at(0).tensor();
      auto weights = inputs.at(1).weights();
      auto biases = inputs.at(2).weights();
      OnnxAttrs attrs(node, ctx);
      float alpha = attrs.get("alpha", 1.f);
      float beta = attrs.get("beta", 1.f);
      bool trans_a = attrs.get("transA", false);
      bool trans_b = attrs.get("transB", false);
      std::vector<int> shape = {0};
      if (attrs.count("shape")) {
        shape = attrs.get<std::vector<int>>("shape");
      }
      if (ctx->getOpsetVersion() < 7)
      {
        ASSERT(attrs.get("broadcast", false), ErrorCode::kUNSUPPORTED_NODE);
      }
      ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::Dims dims = tensor_ptr->getDimensions();

      // Note: TRT requires 3D input for FC layers, so we expand the dims
      bool need_to_expand_dims = (dims.nbDims == 2);
      if (need_to_expand_dims)
      {
        nvinfer1::Dims4 new_shape(dims.d[0], dims.d[1], 1, 1);
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape, GenLayerName(node, "Reshape"));
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
      }
      ASSERT(dims.nbDims >= 4, ErrorCode::kUNSUPPORTED_NODE); //
      int ninput = (dims.nbDims == 3) ? (dims.d[0] * dims.d[1] * dims.d[2]) : (dims.d[1] * dims.d[2] * dims.d[3]);
      ASSERT(!trans_a, ErrorCode::kUNSUPPORTED_NODE);
      if (!trans_b)
      {
        auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
        ASSERT(transposeWeights(weights, {1, 0}, &new_weights, ctx),
              ErrorCode::kUNSUPPORTED_NODE);
        weights = new_weights;
      }

      ASSERT(weights.shape.d[1] == ninput, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(alpha == 1.f, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(beta == 1.f, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(biases.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
      int nrow = biases.shape.d[0];
      ASSERT(weights.shape.d[0] == nrow, ErrorCode::kINVALID_NODE);

      Weights kernelWeights(weights);
      Weights biasWeights(biases);

      bool check_gemvint8 = (kernelWeights.count < 128 || (dims.nbDims > 2 && dims.d[2] > 1));

      if (!check_gemvint8) {
        std::vector<nvinfer1::ITensor *> tensors{tensor_ptr};
        IPluginV2 *plugin = new GemvInt8PluginV2(nrow, kernelWeights, biasWeights, shape);
        auto layer = ctx->network()->addPluginV2(tensors.data(), 1, *plugin);

        layer->setName(GenLayerName(node, "PluginV2.GemvInt8").c_str());
        return {{layer->getOutput(0)}};
      }
    }

    // use DLA fully connect of ours
    if (nGEMV == 1) {
      // Note: Currently this only supports A=tensor, B=weights, C=biases
      ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::ITensor *tensor_ptr = &inputs.at(0).tensor();
      auto weights = inputs.at(1).weights();
      auto biases = inputs.at(2).weights();
      OnnxAttrs attrs(node, ctx);
      float alpha = attrs.get("alpha", 1.f);
      float beta = attrs.get("beta", 1.f);
      bool trans_a = attrs.get("transA", false);
      bool trans_b = attrs.get("transB", false);
      if (ctx->getOpsetVersion() < 7)
      {
        ASSERT(attrs.get("broadcast", false), ErrorCode::kUNSUPPORTED_NODE);
      }
      ASSERT(weights.shape.nbDims == 2, ErrorCode::kUNSUPPORTED_NODE);
      nvinfer1::Dims dims = tensor_ptr->getDimensions();

      // Note: TRT requires 3D input for FC layers, so we expand the dims
      bool need_to_expand_dims = (dims.nbDims == 2);
      if (need_to_expand_dims)
      {
        nvinfer1::Dims4 new_shape(dims.d[0], dims.d[1], 1, 1);
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape, GenLayerName(node, "Reshape"));
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
        dims = tensor_ptr->getDimensions();
      }

      ASSERT(dims.nbDims >= 4, ErrorCode::kUNSUPPORTED_NODE); //
      int ninput = (dims.nbDims == 3) ? (dims.d[0] * dims.d[1] * dims.d[2]) : (dims.d[1] * dims.d[2] * dims.d[3]);
      ASSERT(!trans_a, ErrorCode::kUNSUPPORTED_NODE);
      if (!trans_b)
      {
        auto new_weights = ctx->createTempWeights(weights.type, weights.shape);
        ASSERT(transposeWeights(weights, {1, 0}, &new_weights, ctx),
              ErrorCode::kUNSUPPORTED_NODE);
        weights = new_weights;
      }
      ASSERT(weights.shape.d[1] == ninput, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(alpha == 1.f, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(beta == 1.f, ErrorCode::kUNSUPPORTED_NODE);
      ASSERT(biases.shape.nbDims == 1, ErrorCode::kUNSUPPORTED_NODE);
      int nrow = biases.shape.d[0];
      ASSERT(weights.shape.d[0] == nrow, ErrorCode::kINVALID_NODE);

      auto *layer =
          ctx->network()->addFullyConnected(*tensor_ptr, nrow, weights, biases);
      ASSERT(layer, ErrorCode::kUNSUPPORTED_NODE);
      layer->setName(GenLayerName(node, "FC").c_str());
      tensor_ptr = layer->getOutput(0);
      dims = tensor_ptr->getDimensions();

      // Un-expand the dims back to the original shape
      if (need_to_expand_dims)
      {
        nvinfer1::Dims new_shape{1, {dims.d[0]}};
        tensor_ptr = reshape_tensor(ctx, *tensor_ptr, new_shape, GenLayerName(node, "Reshape2]"));
        ASSERT(tensor_ptr, ErrorCode::kUNSUPPORTED_NODE);
      }

      return {{tensor_ptr}};
    }

    // use Gemm of onnx-tensorrt 8.4
    OnnxAttrs attrs(node, ctx);
    float alpha = attrs.get("alpha", 1.f);
    float beta = attrs.get("beta", 1.f);
    bool transA = attrs.get("transA", false);
    bool transB = attrs.get("transB", false);
    nvinfer1::ITensor& inputA = convertToTensor(inputs.at(0), ctx);
    // Validate inputs
    ASSERT(inputs.at(0).shape().nbDims == 2 && inputs.at(1).shape().nbDims == 2 && "GEMM must have 2D inputs!", ErrorCode::kINVALID_NODE);
    // TRT does not support INT32 input types for this node
    ASSERT(!inputs.at(0).isInt32() && !inputs.at(1).isInt32() && "TensorRT doesn't support INT32 inputs for GEMM!",
        ErrorCode::kUNSUPPORTED_NODE);

    nvinfer1::ITensor* inputB{nullptr};

    if (inputs.at(1).is_weights())
    {
        ShapedWeights weights = inputs.at(1).weights();
        nvinfer1::IConstantLayer* weightsLayer
            = ctx->network()->addConstant(weights.shape, static_cast<nvinfer1::Weights>(weights));
        // Map the constant layer to the weights name.
        ctx->registerLayer(weightsLayer, node.input(1));
        ctx->network()->setWeightsName(weights, weights.getName());
        inputB = weightsLayer->getOutput(0);
    }
    else
    {
        inputB = &inputs.at(1).tensor();
    }

    nvinfer1::ITensor* inputASqueezed = &inputA;
    nvinfer1::Dims newDims = squeeze_trailing_dims(inputA.getDimensions());
    // When A has more than 2 dimensions, it needs to be flattened.
    if (newDims.nbDims > 2)
    {
        newDims = nvinfer1::Dims{1, {-1}};
    }
    // Due to other TRT layers, inputA may sometimes have trailing 1s that need to be removed.
    if (newDims.nbDims < inputA.getDimensions().nbDims)
    {
        nvinfer1::IShuffleLayer* squeeze = ctx->network()->addShuffle(inputA);
        squeeze->setReshapeDimensions(newDims);
        squeeze->setZeroIsPlaceholder(false);
        inputASqueezed = squeeze->getOutput(0);
    }

    const auto getMatrixOp = [](const nvinfer1::ITensor& input, bool transpose) {
        if (input.getDimensions().nbDims == 1)
        {
            return nvinfer1::MatrixOperation::kVECTOR;
        }
        if (transpose)
        {
            return nvinfer1::MatrixOperation::kTRANSPOSE;
        }
        return nvinfer1::MatrixOperation::kNONE;
    };

    nvinfer1::MatrixOperation opA = getMatrixOp(*inputASqueezed, transA);
    nvinfer1::MatrixOperation opB = getMatrixOp(*inputB, transB);

    LOG_VERBOSE("Using opA: " << static_cast<int>(opA) << " opB: " << static_cast<int>(opB));
    LOG_VERBOSE("GEMM: A, after squeezing: " << inputASqueezed->getDimensions());

    nvinfer1::IMatrixMultiplyLayer* matmul = ctx->network()->addMatrixMultiply(*inputASqueezed, opA, *inputB, opB);
    ctx->registerLayer(matmul, getNodeName(node));
    nvinfer1::ITensor* matmulTensor = matmul->getOutput(0);

    // Scale A*B if needed.
    if (alpha != 1.f)
    {
        nvinfer1::IConstantLayer* alphaConstant
            = addConstantScalar(ctx, alpha, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        nvinfer1::ITensor* alphaConstantTensor = alphaConstant->getOutput(0);
        CHECK(broadcastTensors(ctx, alphaConstantTensor, matmulTensor));
        nvinfer1::IElementWiseLayer* scaledMatmul = ctx->network()->addElementWise(
            *alphaConstantTensor, *matmulTensor, nvinfer1::ElementWiseOperation::kPROD);
        matmulTensor = scaledMatmul->getOutput(0);
    }

    // In opset 11, the bias tensor is an optional input
    if (inputs.size() > 2)
    {
        nvinfer1::ITensor* biasTensor = &convertToTensor(inputs.at(2), ctx);

        // Scale C if needed
        if (beta != 1.f)
        {
            nvinfer1::IConstantLayer* betaConstant
                = addConstantScalar(ctx, beta, ::ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            nvinfer1::ITensor* betaConstantTensor = betaConstant->getOutput(0);
            CHECK(broadcastTensors(ctx, betaConstantTensor, biasTensor));
            nvinfer1::IElementWiseLayer* scaledBias = ctx->network()->addElementWise(
                *betaConstantTensor, *biasTensor, nvinfer1::ElementWiseOperation::kPROD);
            biasTensor = scaledBias->getOutput(0);
        }
        // A*B may be lower rank than C in TRT, so need to squeeze C.
        if (ctx->getOpsetVersion() < 7 && !attrs.get("broadcast", false))
        {
            nvinfer1::Dims squeezeDims = squeeze_leading_dims(biasTensor->getDimensions());
            biasTensor = reshapeTensor(ctx, *biasTensor, squeezeDims);
        }
        CHECK(broadcastTensors(ctx, matmulTensor, biasTensor));
        nvinfer1::IElementWiseLayer* biasAdd
            = ctx->network()->addElementWise(*matmulTensor, *biasTensor, nvinfer1::ElementWiseOperation::kSUM);
        return {{biasAdd->getOutput(0)}};
    }

    return {{matmulTensor}};
}

// Batch Pad Concatenate op for main/narrow combined onnx
DEFINE_BUILTIN_OP_IMPORTER(ConvLSTM) {
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

  std::vector<int> attrv;
  OnnxAttrs attrs(node, ctx);
  {
    attrv.push_back(attrs.get<int>("in_c", 512));
    attrv.push_back(attrs.get<int>("hidden_c", 512));
    attrv.push_back(attrs.get<int>("kernel", 3));
    attrv.push_back(attrs.get<int>("repeat", 1));
    attrv.push_back(attrs.get<int>("stride", 1));
    attrv.push_back(attrs.get<int>("pad", 1));
  }

  // Add plugin layer
#if 0  // mode 0, all processing in one plugin
  ConvLSTMPlugin *pPlugin = new ConvLSTMPlugin(weights, attrv);
  auto *layer = ctx->addPlugin(pPlugin, tensors);
#else  // 1. concat + conv(trt) + 3. other( sigmod & tanh & mul )
  int magicNum = rand() % 0x7FFFFFFF;
  attrv[3] = magicNum;
  //ConvLSTMPlugin *pPlugin = new ConvLSTMPlugin(1, attrv);
  //auto *layer1 = ctx->addPlugin(pPlugin, tensors);
  IPluginV2 *pPlugin = new ConvLSTMPluginV2(1, attrv);
  auto layer1 = ctx->network()->addPluginV2(tensors.data(), 1, *pPlugin);
  nvinfer1::ITensor *tensor_ptr = layer1->getOutput(0);
  layer1->setName(GenLayerName(node, "ConvLSTM.Concat").c_str());
  tensor_ptr->setName((node.output(0) + ".ConvLSTM.Concat.Out").c_str());

  {
    ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(inputs.at(1).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    auto kernel_weights = inputs.at(1).weights();
    nvinfer1::Dims dims = tensor_ptr->getDimensions();
    ASSERT(dims.nbDims >= 3, ErrorCode::kUNSUPPORTED_NODE);
    ASSERT(kernel_weights.shape.nbDims == 4, ErrorCode::kUNSUPPORTED_NODE);
    nvinfer1::Weights bias_weights;
    if (inputs.size() == 3) {
      ASSERT(inputs.at(2).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
      auto shaped_bias_weights = inputs.at(2).weights();
      ASSERT(shaped_bias_weights.shape.nbDims == 1, ErrorCode::kINVALID_NODE);
      ASSERT(shaped_bias_weights.shape.d[0] == kernel_weights.shape.d[0], ErrorCode::kINVALID_NODE);
      bias_weights = shaped_bias_weights;
    } else {
      bias_weights = ShapedWeights::empty(kernel_weights.type);
    }

    nvinfer1::DimsHW kernel_size(attrv[2], attrv[2]);
    nvinfer1::DimsHW strides(attrv[4], attrv[4]);
    nvinfer1::DimsHW beg_padding(attrv[5], attrv[5]);
    nvinfer1::DimsHW dilations(1, 1);

    ASSERT(kernel_size.h() == kernel_weights.shape.d[2], ErrorCode::kINVALID_NODE);
    ASSERT(kernel_size.w() == kernel_weights.shape.d[3], ErrorCode::kINVALID_NODE);
    int noutput = kernel_weights.shape.d[0];  // Note: Weights order is KCRS

    auto layer2 = ctx->network()->addConvolution(*tensor_ptr, noutput, kernel_size, kernel_weights, bias_weights);
    ASSERT(layer2, ErrorCode::kUNSUPPORTED_NODE);
    layer2->setName(GenLayerName(node, "ConvLSTM.Conv").c_str());
    layer2->setStride(strides);
    layer2->setPadding(beg_padding);
    layer2->setDilation(dilations);
    layer2->setNbGroups(1);
    tensor_ptr = layer2->getOutput(0);
    tensor_ptr->setName((node.output(0) + ".ConvLSTM.Conv.Out").c_str());
  }

  //pPlugin = new ConvLSTMPlugin(3, attrv);
  //auto *layer = ctx->addPlugin(pPlugin, {tensor_ptr});
  std::vector<nvinfer1::ITensor *> tensors2{tensor_ptr};
  pPlugin = new ConvLSTMPluginV2(3, attrv);
  auto *layer = ctx->network()->addPluginV2(tensors2.data(), 1, *pPlugin);
#endif
  // Return layer output
  RETURN_FIRST_OUTPUT(layer);
}

// ConfidenceFilter for MOD/SOD/RSM/KPTL output
DEFINE_BUILTIN_OP_IMPORTER(ConfidenceFilter) {
  ASSERT(inputs.size() >= 1, ErrorCode::kINVALID_NODE);
  std::vector<nvinfer1::ITensor *> tensors;
  for (auto &input : inputs) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensors.push_back(&input.tensor());
  }

  OnnxAttrs attrs(node, ctx);
  auto max_num = attrs.get<int>("max_num", 128);
  // auto conf_offset = attrs.get<int>("conf_offset", 1);
  std::vector<int> conf_offsets = attrs.get<std::vector<int>>("conf_offset");

  // auto threshold = attrs.get<float>("conf_threshold", 0.7);
  std::vector<float> conf_threshs = attrs.get<std::vector<float>>("conf_threshold");
  float threshold = 0;
  if (conf_threshs.size() > 1) {  
    //ASSERT(conf_threshs.size() == 3, ErrorCode::kUNSUPPORTED_NODE);
    threshold = conf_threshs[0];
  } else {
    threshold = attrs.get<float>("conf_threshold", 0.7);
  }

  int conf_offset = 0;
  if (conf_offsets.size() > 1) {  
    //ASSERT(conf_offsets.size() == 3, ErrorCode::kUNSUPPORTED_NODE);
    for (unsigned int i = conf_threshs.size(); i < conf_offsets.size(); i++) {
      conf_threshs.push_back(threshold);
    }
  } else {
    conf_offset = attrs.get<int>("conf_offset", 1);
    conf_threshs.resize(1);
    conf_threshs[0] = threshold;
    conf_offsets.resize(1);
    conf_offsets[0] = conf_offset;
  }

  auto mode_s = attrs.get<std::string>("mode", "sigmoid");  // sigmod
  std::vector<int> fpn_shape(3, 0);
  if (attrs.count("fpn_shape")) {
    fpn_shape = attrs.get<std::vector<int>>("fpn_shape");
    ASSERT(fpn_shape.size() == 3, ErrorCode::kUNSUPPORTED_NODE);
  }
  std::vector<int> fpn_type;
  if (attrs.count("fpn_type")) {
    fpn_type = attrs.get<std::vector<int>>("fpn_type");
  }

  {
    char *val = getenv("TRT_FILTER_MAXNUM");
    if (NULL != val) {
      max_num = atoi(val);
      DPRINTF(2, "getenv TRT_FILTER_MAXNUM = %d\n", max_num);
    }
  }

  // int cf_version = 0;
  // cf_version = attrs.get<int>("version", 0);
  // char *val = getenv("TRT_CF");  // use upsample plugin
  // if (NULL != val) {
  //   cf_version = atoi(val);
  // }

  // if(1 == cf_version) {
  //   auto *plugin = new ConfidenceFilterV2Plugin(max_num, conf_offset, threshold, mode_s, fpn_shape, fpn_type);
  //   auto layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
  //   layer->setName(GenLayerName(node, "PluginV2.ConfidenceFilterV2").c_str());

  //   // Return layer output
  //   return {{layer->getOutput(0)}};
  // }
  // Add plugin layer
  IPluginV2 *plugin;
  // if (conf_offsets.size() > 1 && conf_threshs.size() > 1) {  // only for bev model
    plugin = new ConfidenceFilterV2(max_num, conf_offsets, conf_threshs, mode_s, fpn_shape, fpn_type);

  // } else {
  //   plugin = new ConfidenceFilterV2(max_num, conf_offset, threshold, mode_s, fpn_shape, fpn_type);
  // }
  auto layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
  layer->setName(GenLayerName(node, "PluginV2.ConfidenceFilter").c_str());
  // Return layer output
  return {{layer->getOutput(0)}};
}

#include "YhatsFilter.h"
// Yhats Filter
DEFINE_BUILTIN_OP_IMPORTER(YhatsFilter) {
  ASSERT(inputs.size() >= 1, ErrorCode::kINVALID_NODE);
  std::vector<nvinfer1::ITensor *> tensors;
  for (auto &input : inputs) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensors.push_back(&input.tensor());
  }

  OnnxAttrs attrs(node, ctx);
  auto channel   = attrs.get<int>("channel", 8);
  auto height    = attrs.get<int>("height",  128);
  auto width     = attrs.get<int>("width",   320);
  auto radius    = attrs.get<int>("radius",  3);
  auto choffset  = attrs.get<int>("choffset",  0);
  auto offset    = attrs.get<float>("offset",  112);

  // Add plugin layer
  IPluginV2 *plugin;
  plugin = new YhatsFilter(channel, height, width, radius, choffset, offset);
  
  auto layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
  layer->setName(GenLayerName(node, "PluginV2.YhatsFilter").c_str());
  // Return layer output
  return {{layer->getOutput(0)}};
}

#include "Res2ChannelPad.h"
DEFINE_BUILTIN_OP_IMPORTER(Res2ChannelPad) {
  ASSERT(inputs.size() >= 1, ErrorCode::kINVALID_NODE);
  std::vector<nvinfer1::ITensor *> tensors;
  for (auto &input : inputs) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensors.push_back(&input.tensor());
  }

  OnnxAttrs attrs(node, ctx);
  auto resolution    = attrs.get<int>("resolution",   2);
  auto rowPadding    = attrs.get<int>("rowPadding",   0);
  auto columnPadding = attrs.get<int>("columnPadding",0);


  // Add plugin layer
  IPluginV2 *plugin;
  plugin = new Res2ChannelPad((uint32_t)resolution, (uint32_t)rowPadding, (uint32_t)columnPadding);
  
  auto layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
  layer->setName(GenLayerName(node, "PluginV2.Res2ChannelPad").c_str());
  // Return layer output
  return {{layer->getOutput(0)}};
}

//Channel2Spatial for mixed side -------------------------------------------------------------------------------------------
#include "Channel2spatial.h"
DEFINE_BUILTIN_OP_IMPORTER(Channel2Spatial) {
  ASSERT(inputs.size() == 1, ErrorCode::kINVALID_NODE);
  std::vector<nvinfer1::ITensor *> tensors;
  for (auto &input : inputs) {
    ASSERT(input.is_tensor(), ErrorCode::kUNSUPPORTED_NODE);
    tensors.push_back(&input.tensor());
  }

  OnnxAttrs attrs(node, ctx);
  auto scale    = attrs.get<int>("scale", 1);
  auto mixed    = attrs.get<int>("mixed_batch", 0);

  IPluginV2 *plugin = (IPluginV2*) new Channel2Spatial(scale, mixed);
  auto layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
  layer->setName(GenLayerName(node, "PluginV2.Channel2Spatial").c_str());
  // Return layer output
  return {{layer->getOutput(0)}};

}
//#endif  // NV_TENSORRT_MAJOR >= 6
