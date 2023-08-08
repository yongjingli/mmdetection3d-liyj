#include <chrono>
#include <iostream>
#include <memory>
#include "dms_pipeline.h"
#include "utils.h"

int main(int argc, char **argvs) {
  int engine_id = 0;
  std::string dla_lib_path = "./libonnxtrt_dla.so";
  std::string fd_engien_path = "./faceDetection_fp16.dla";
  std::string lm_engien_path = "./landmark_fp16.dla";
  int fd_engine_id = 0;
  int lm_engine_id = 2;
  std::string input_data_path = "./20190402202013_-_DA0_rec_extract_frame_cosine_00001131_transform_output.bin";
  std::unique_ptr<DMSPipeline> dms_tool = std::make_unique<DMSPipeline>(dla_lib_path);

  if (!dms_tool->CreatEngines(fd_engine_id, fd_engien_path, lm_engine_id, lm_engien_path)) return 0;

  uint8_t *src_gpu;
  CUDA_CHECK(cudaMalloc((void **)&src_gpu, g_image_h * g_image_w * sizeof(uint8_t) * 2));
  char *inputStream = new char[g_image_h * g_image_w];
  if (!readBinToBuffer(input_data_path, inputStream, g_image_h * g_image_w)) return 0;
  CUDA_CHECK(cudaMemcpy(src_gpu, inputStream, g_image_h * g_image_w * sizeof(uint8_t), cudaMemcpyHostToDevice));

  Face output_face = dms_tool->ExtractFaceBox(src_gpu, fd_engine_id);
  std::cout << output_face.score << " " << output_face.xmin << " " << output_face.ymin << " " << output_face.xmax << " "
            << output_face.ymax << std::endl;

  DrawBboxGray((uint8_t *)inputStream, g_image_w, g_image_h, (int)output_face.xmin, (int)output_face.ymin,
               (int)output_face.xmax, (int)output_face.ymax);
  const std::string outfile = "./face_detection_output.pgm";
  writePPMGray(inputStream, g_image_w, g_image_h, outfile);

  float *landmarks = new float[g_lm_number * 2];
  dms_tool->ExtractLandmark(src_gpu, output_face, lm_engine_id, landmarks);

  int crop_h = (int)output_face.ymax - (int)output_face.ymin;
  int crop_w = (int)output_face.xmax - (int)output_face.xmin;
  DrawLandmarkGray((uint8_t *)inputStream, landmarks, g_lm_number, g_image_w, g_image_h, (int)output_face.xmin,
                   (int)output_face.ymin, crop_h, crop_w);
  const std::string outfile2 = "./landmark_output.pgm";
  writePPMGray(inputStream, g_image_w, g_image_h, outfile2);

  // free engine and data
  dms_tool->DestoryEngines(fd_engine_id, lm_engine_id);
  delete[] inputStream;
  delete[] landmarks;
  cudaFree(src_gpu);
}