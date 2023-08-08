#include <iostream>
#include "FusedMultiHeadAttention.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <array>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
// #include "helper.h"
// #include "linear_combination_exp.h"
#include "cutlass/epilogue/thread/linear_combination_exp.h"

#define _CUDA_TIME_POINT_

#include "cuda_runtime.h"

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }


using ElementAccumulator_FP16 = cutlass::half_t;   
using ElementComputeEpilogue = ElementAccumulator_FP16; 
// The code section below describes matrix layout of input and output matrices. Column Major for
// output = exp( A * B ) * C
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutputAB = cutlass::layout::ColumnMajor;
using LayoutInputC = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using ShapeMMAThreadBlock_0 = cutlass::gemm::GemmShape<128, 128, 32>;  
using ShapeMMAWarp_0 = cutlass::gemm::GemmShape<64, 64, 32>; 
using ShapeMMAOp_0 = cutlass::gemm::GemmShape<16, 8, 8>; 

using ShapeMMAThreadBlock_1 = cutlass::gemm::GemmShape<128, 128, 32>;  
using ShapeMMAWarp_1 = cutlass::gemm::GemmShape<64, 64, 32>;  
using ShapeMMAOp_1 = cutlass::gemm::GemmShape<16, 8, 16>;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
constexpr int AlignmentConstraint = 8;
constexpr int NumStage_0 = 3;
constexpr int NumStage_1 = 5;

using Gemm_0_FP16 = cutlass::gemm::device::Gemm<
ElementAccumulator_FP16, LayoutInputB,   // transposed B operand
ElementAccumulator_FP16, LayoutInputA,   // transposed A operand
ElementAccumulator_FP16, LayoutOutputAB, // transposed AB operand
ElementAccumulator_FP16,
MMAOp,
SmArch,
ShapeMMAThreadBlock_0,
ShapeMMAWarp_0,
ShapeMMAOp_0,
cutlass::epilogue::thread::LinearCombinationExpSoftmax<
  ElementAccumulator_FP16,
  AlignmentConstraint,
  ElementAccumulator_FP16,
  ElementAccumulator_FP16
>,
SwizzleThreadBlock,
NumStage_0
>;

using Gemm_1_FP16 = cutlass::gemm::device::Gemm<
ElementAccumulator_FP16, LayoutInputC, // transposed C operand
ElementAccumulator_FP16, LayoutOutputAB, // transposed AB operand
ElementAccumulator_FP16, LayoutOutput, // transposed output operand
ElementAccumulator_FP16,
MMAOp,
SmArch,
ShapeMMAThreadBlock_1,
ShapeMMAWarp_1,
ShapeMMAOp_1,
cutlass::epilogue::thread::LinearCombination<
  ElementAccumulator_FP16,
  AlignmentConstraint,
  ElementAccumulator_FP16,
  ElementAccumulator_FP16
>,
SwizzleThreadBlock,
NumStage_1
>;

// Static class fields initialization
PluginFieldCollection FusedMultiHeadAttentionCreator::mFC{};
std::vector<PluginField> FusedMultiHeadAttentionCreator::mPluginAttributes;


int FusedMultiHeadAttentionForward_fp16(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
        float scale, std::vector<std::vector<int>>& inputDims, cudaStream_t stream) {
#ifdef _CUDA_TIME_POINT_
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    // if (stream == nullptr) return -1;
    assert(workspace  != nullptr);
    int head = inputDims[0][0];
    int m = inputDims[0][1];
    int k = inputDims[0][2];
    int n = inputDims[1][2];
    int L = inputDims[2][2];
    head = head * batchSize;

    long int lda_ = k; 
    long int ldb_ = n;
    long int ldc_ = k;
    long int ldab_ = n;
    long int ldout_ = L;

    int tensor_stride_a_ = m * k;
    int tensor_stride_b_ = k * n;
    int tensor_stride_c_ = n * L;
    int tensor_stride_ab_ = m * n;
    int tensor_stride_out_ = m * L;

    cutlass::Status status;
    int split_k_slices_ = 1;

    std::vector<Gemm_0_FP16> gemm_ops_0_;
    std::vector<Gemm_1_FP16> gemm_ops_1_;
    gemm_ops_0_.resize(head);
    gemm_ops_1_.resize(head);
    ElementAccumulator_FP16* tensor_a_ = (ElementAccumulator_FP16*)inputs[0];
    ElementAccumulator_FP16* tensor_b_ = (ElementAccumulator_FP16*)inputs[1];
    ElementAccumulator_FP16* tensor_c_ = (ElementAccumulator_FP16*)inputs[2];
    // ElementAccumulator_FP16* tensor_exp_max_ = (ElementAccumulator_FP16*)inputs[3];
    ElementAccumulator_FP16* tensor_out_ = (ElementAccumulator_FP16*)outputs[0];
    ElementAccumulator_FP16* tensor_ab_ = (ElementAccumulator_FP16*)workspace;

    // ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    // ElementComputeEpilogue beta = ElementComputeEpilogue(-scale);
    for (int i=0; i<head; ++i) {
        typename Gemm_0_FP16::Arguments args_0{{n, m, k},
            {tensor_b_ + i * tensor_stride_b_, ldb_},
            {tensor_a_ + i * tensor_stride_a_, lda_},
            {tensor_ab_ , ldab_},
            {tensor_ab_ , ldab_},
            {cutlass::half_t(1), cutlass::half_t(-scale)},
            split_k_slices_};
        
        status = gemm_ops_0_[i].initialize(args_0);
        CUTLASS_CHECK(status);
        status = gemm_ops_0_[i](stream);
        CUTLASS_CHECK(status);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        typename Gemm_1_FP16::Arguments args_1{{L, m, n},
            {tensor_c_ + i * tensor_stride_c_, ldc_},          
            {tensor_ab_ , ldab_},
            {tensor_out_ + i * tensor_stride_out_, ldout_},
            {tensor_out_ + i * tensor_stride_out_, ldout_},
            {cutlass::half_t(1), cutlass::half_t(0)},
            split_k_slices_};
        status = gemm_ops_1_[i].initialize(args_1);
        CUTLASS_CHECK(status);
        status = gemm_ops_1_[i](stream);
        CUTLASS_CHECK(status);  
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef _CUDA_TIME_POINT_
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    float milliseconds = elapsedTime;
    printf("Total time: %f ms. Average time: %f ms.\n", elapsedTime, milliseconds);
    printf("TOPS: %.2f\n", (((double)m * n * k * 2 * (double)(1+3/2)) / (milliseconds / 1000.)) / 1e12);
#endif

    return 0;
}

