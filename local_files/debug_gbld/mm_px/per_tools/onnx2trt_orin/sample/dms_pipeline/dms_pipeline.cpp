
#include "dms_pipeline.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

DMSPipeline::DMSPipeline(const std::string &dla_lib_path) {
  infer_tool_ = std::make_unique<DLAInferTool>(dla_lib_path);
  face_detection_post_ = std::make_unique<FaceDetection>();
  landmark_post_ = std::make_unique<Landmark>();
}

bool DMSPipeline::CreatEngines(int fd_engine_id, const std::string &fd_engine_path, int lm_engine_id,
                               const std::string &lm_engine_path) {
  // creat face detection engine
  ENGINE_STATUS status = infer_tool_->CreatEngine(fd_engine_path, fd_engine_id);
  if (status != CREATED) {
    std::cerr << "ERROR face detection engine Init failed !!!" << std::endl;
    return false;
  }
  status = infer_tool_->GetInOutBuffer(&pstream_fd_, fd_engine_id, fd_input_buffer_, fd_in_num_, fd_in_size_,
                                       fd_output_buffer_, fd_out_num_, fd_out_size_, fd_out_size_sum_);
  if (status == RUNFAILED) {
    std::cerr << "ERROR face detection GetInOutBuffer failed !!!" << std::endl;
    return false;
  }
  dms_isp_fd_ = std::make_unique<DmsDetectionISP>(g_image_h, g_image_w, g_fd_input_h, g_fd_input_w, pstream_fd_);
  // creat landmark engine
  status = infer_tool_->CreatEngine(lm_engine_path, lm_engine_id);
  if (status != CREATED) {
    std::cerr << "ERROR face landmark engine Init failed !!!" << std::endl;
    return false;
  }
  status = infer_tool_->GetInOutBuffer(&pstream_lm_, lm_engine_id, lm_input_buffer_, lm_in_num_, lm_in_size_,
                                       lm_output_buffer_, lm_out_num_, lm_out_size_, lm_out_size_sum_);

  if (status == RUNFAILED) {
    std::cerr << "ERROR face landmark GetInOutBuffer failed !!!" << std::endl;
    return false;
  }
  dms_isp_lm_ = std::make_unique<DmsLandmarkISP>(g_image_h, g_image_w, g_lm_input_h, g_lm_input_w, pstream_lm_);
  return true;
}
void DMSPipeline::DestoryEngines(int fd_engine_id, int lm_engine_id) {
  infer_tool_->DestoryEngine(fd_engine_id);
  infer_tool_->DestoryEngine(lm_engine_id);
}

Face DMSPipeline::ExtractFaceBox(const uint8_t *input_gpu, int fd_engine_id) {
  int onein_offset = g_fd_input_h * g_fd_input_w * 16 * sizeof(uint16_t);  // Byte
  int batch = fd_in_size_[0] / onein_offset / 2;
  int oneout_offset = fd_out_size_[0] / batch / 2 / sizeof(uint16_t);  // fp16
  std::cout << "ExtractFaceBox " << batch << "x[3," << g_fd_input_h << "," << g_fd_input_w << "] = " << fd_in_size_[0]
            << "Byte" << std::endl;
  std::cout << "onein_offset=" << onein_offset << " oneout_offset=" << oneout_offset << std::endl;
  int fd_in_test_offset = 0;
  {
    char *val = getenv("FD_TESTOFFSET");
    if (NULL != val) {
      fd_in_test_offset = atoi(val);
      std::cout << "fd_in_test_offset=" << fd_in_test_offset << std::endl;
    }
  }

  // preprocess
  for (int ib = 0; ib < batch; ib++) {
    char *input_buf = (char *)fd_input_buffer_[0] + onein_offset * ib;
    dms_isp_fd_->resizeAndToCHW16Half(input_gpu + fd_in_test_offset * ib, g_image_h, g_image_w, g_fd_input_h,
                                      g_fd_input_w, input_buf);
  }

  // run inference
  ENGINE_STATUS status =
      infer_tool_->inference((char *)fd_input_buffer_[0], fd_engine_id, (char *)fd_output_buffer_[0]);
  if (status != RUNWELL) {
    std::cerr << "ERROR face detection inference failed !!!" << std::endl;
    Face tmp_face{0, 0, 0, 0, 0, 0};
    return tmp_face;
  }
  // postprocess
  Face output_faces[batch];
  for (int ib = 0; ib < batch; ib++) {
    uint16_t *output_buf[1] = {((uint16_t **)fd_output_buffer_)[0] + oneout_offset * ib};
    Face output_face = face_detection_post_->ExtractFace(output_buf, fd_out_num_);
    std::cout << "B" << ib << ":" << fd_out_num_ << " " << output_face.score << " " << output_face.xmin << " "
              << output_face.ymin << " " << output_face.xmax << " " << output_face.ymax << std::endl;
    output_faces[ib] = output_face;
  }
  // return the output
  return output_faces[0];
}

void DMSPipeline::ExtractLandmark(const uint8_t *input_gpu, const Face &crop_face, int lm_engine_id, float *output) {
  int onein_offset = g_lm_input_h * g_lm_input_w * 16 * sizeof(uint16_t);  // Byte
  int batch = lm_in_size_[0] / onein_offset / 2;
  int oneout_offset[lm_out_num_];  // fp16

  std::cout << "ExtractLandmark " << batch << "x[3," << g_lm_input_h << "," << g_lm_input_w << "] = " << lm_in_size_[0]
            << "Byte" << std::endl;
  std::cout << "onein_offset=" << onein_offset << " oneout_offset=";
  for (int i = 0; i < lm_out_num_; i++) {
    oneout_offset[i] = lm_out_size_[i] / batch / sizeof(uint16_t);
    std::cout << oneout_offset[i] << ", ";
  }
  std::cout << std::endl;
  float lm_in_test_offset = 0;
  {
    char *val = getenv("LM_TESTOFFSET");
    if (NULL != val) {
      lm_in_test_offset = atof(val);
      std::cout << "lm_in_test_offset=" << lm_in_test_offset << std::endl;
    }
  }

  // preprocess
  for (int ib = 0; ib < batch; ib++) {
    char *input_buf = (char *)lm_input_buffer_[0] + onein_offset * ib;
    dms_isp_lm_->CropResizeAndToCHW16Half(input_gpu, input_buf, g_image_h, g_image_w, g_lm_input_h, g_lm_input_w,
                                          crop_face.xmin + lm_in_test_offset * ib, crop_face.ymin, crop_face.xmax,
                                          crop_face.ymax);
  }

  char tmp_out_buffer[lm_out_size_sum_];
  // run inference
  ENGINE_STATUS status = infer_tool_->inference((char *)lm_input_buffer_[0], lm_engine_id, tmp_out_buffer);
  if (status != RUNWELL) {
    std::cerr << "ERROR face landmark inference failed !!!" << std::endl;
    return;
  }

  // postprocess
  for (int ib = 0; ib < batch; ib++) {
    uint16_t *output_buf[lm_out_num_];
    for (int i = 0; i < lm_out_num_; i++) {
      output_buf[i] = ((uint16_t **)lm_output_buffer_)[i] + oneout_offset[i] * ib;
    }
    landmark_post_->ExtractLandmark(output_buf, 0, output);
    std::cout << "B" << ib << ":" << output[0] << "," << output[1] << "," << output[2] << "," << output[2] << std::endl;
  }
}
