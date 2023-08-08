#ifndef DMS_PIPELINE_H_
#define DMS_PIPELINE_H_
#include <memory>
#include "config.h"
#include "dms_cuda_utils.h"
#include "face_detection.h"
#include "inference_tool.h"
#include "landmark.h"
class DMSPipeline {
 private:
  /* data */
  std::unique_ptr<DLAInferTool> infer_tool_;
  std::unique_ptr<DmsDetectionISP> dms_isp_fd_;
  std::unique_ptr<DmsLandmarkISP> dms_isp_lm_;
  std::unique_ptr<FaceDetection> face_detection_post_;
  std::unique_ptr<Landmark> landmark_post_;
  /*for face detection */
  void *fd_input_buffer_[MAX_IN_BUFFER];
  void *fd_output_buffer_[MAX_OUT_BUFFER];
  int fd_in_size_[MAX_IN_BUFFER];
  int fd_out_size_[MAX_OUT_BUFFER];
  int fd_in_num_, fd_out_num_, fd_out_size_sum_;
  /*for  landmark */
  void *lm_input_buffer_[MAX_IN_BUFFER];
  void *lm_output_buffer_[MAX_OUT_BUFFER];
  int lm_in_size_[MAX_IN_BUFFER];
  int lm_out_size_[MAX_OUT_BUFFER];
  int lm_in_num_, lm_out_num_, lm_out_size_sum_;
  /*for nvscisync*/
  void *pstream_fd_{nullptr};
  void *pstream_lm_{nullptr};

 public:
  DMSPipeline(const std::string &dla_lib_path);

  void ExtractLandmark(const uint8_t *input_gpu, const Face &crop_face, int lm_engine_id, float *output);
  Face ExtractFaceBox(const uint8_t *input_gpu, int fd_engine_id);
  void DestoryEngines(int fd_engine_id, int lm_engine_id);
  bool CreatEngines(int fd_engine_id, const std::string &fd_engine_path, int lm_engine_id,
                    const std::string &lm_engine_path);
  ~DMSPipeline(){};
};

#endif  // DMS_PIPELINE_H_