#ifndef DMS_CUDA_UTILS_H_
#define DMS_CUDA_UTILS_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppdefs.h>
#include <nppi_geometry_transforms.h>
#include <iostream>
#define CUDA_CHECK(condition)                                                      \
  do {                                                                             \
    cudaError_t error = condition;                                                 \
    if (error != cudaSuccess) std::cout << cudaGetErrorString(error) << std::endl; \
  } while (0)

class DmsISPBase {
 protected:
  int img_height_, img_width_, input_h_, input_w_;
  uint8_t *resize_tmp_;
  const int block_size_{256};
  dim3 block_, grid_;
  cudaStream_t stream_;
  NppStreamContext nppStream_;

  DmsISPBase() = delete;
  DmsISPBase(const int img_height, const int img_width, const int input_h, const int input_w, void *pstream);
  ~DmsISPBase();
  void CHWInt8ToCHW16Half(const uint8_t *input_gpu, void *output_gpu);
  // void CHWFp32ToCHW16Half(const uint8_t *input_gpu, int out_h, int out_w, char *output_gpu);
};
class DmsDetectionISP : public DmsISPBase {
 private:
  void Resize(const uint8_t *input_gpu, int in_h, int in_w, int out_h, int out_w);

 public:
  DmsDetectionISP(const int img_height, const int img_width, const int fd_input_h, const int fd_input_w, void *pstream);
  void resizeAndToCHW16Half(const uint8_t *input_gpu, int in_h, int in_w, int out_h, int out_w, void *output_gpu);
};

class DmsLandmarkISP : public DmsISPBase {
 private:
  void CropAndResize(const uint8_t *input_gpu, int in_h, int in_w, int out_h, int out_w, int x1, int y1, int x2,
                     int y2);

 public:
  DmsLandmarkISP(const int img_height, const int img_width, const int lm_input_h, const int lm_input_w, void *pstream);
  void CropResizeAndToCHW16Half(const uint8_t *input_gpu, void *output_cpu, int in_h, int in_w, int out_h, int out_w,
                                int x1, int y1, int x2, int y2);
};
#endif  // DMS_CUDA_UTILS_H_