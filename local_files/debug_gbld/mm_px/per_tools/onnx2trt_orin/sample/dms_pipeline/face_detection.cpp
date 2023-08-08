#include "face_detection.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include "config.h"
bool comp(const Face &a, const Face &b)  // define "<" by myself,for std::sort
{
  return a.score < b.score;
}
void FaceDetection::GenPriorbox() {
  int anchor_num = 0;
#ifdef USE_G3_MODEL  // G3 model
  for (int i = 0; i < g_fd_out_layer_num; ++i) {
    anchor_num += g_fd_output_size[i] * g_fd_output_size[i];
  }

  anchors_.reserve(anchor_num);
  for (int i = 0; i < g_fd_out_layer_num; ++i) {
    for (int h = 0; h < g_fd_output_size[i]; ++h) {
      for (int w = 0; w < g_fd_output_size[i]; ++w) {
#else
  anchor_num = g_fd_output_size[0] * g_fd_output_size[1] * 2;
  anchors_.reserve(anchor_num);
  for (int i = 0; i < 2; ++i) {
    for (int h = 0; h < g_fd_output_size[0]; ++h) {
      for (int w = 0; w < g_fd_output_size[1]; ++w) {
#endif
        // Anchor anchor;
        float s_kx = float(g_fd_min_size[i]) / g_fd_input_w;
        float s_ky = float(g_fd_min_size[i]) / g_fd_input_h;
        float cx = (w + 0.5) * g_fd_steps[i] / g_fd_input_w;
        float cy = (h + 0.5) * g_fd_steps[i] / g_fd_input_h;
        anchors_.emplace_back(cx, cy, s_kx, s_ky);
      }
    }
  }
}
FaceDetection::FaceDetection() {
  this->GenPriorbox();
  filtered_face_.reserve(g_fd_top_k);
  nms_out_face_.reserve(g_fd_keep_top_k);
  h2f_tool = std::make_unique<half2float>();
}
/* simple softmax ,it is a inline function
https://blog.csdn.net/fengbingchun/article/details/75220591 */
inline float FaceDetection::SlimSoftMax(const float face, const float no_face) {
  return face > no_face ? (1 / (1 + expf(no_face - face))) : 0.0f;
}
/*the output format is (H W 16),the [4,6) is the score of face .the [0,2) is the bbox.*/
void FaceDetection::FilterFace(const uint16_t *data, const int bbox_num, const int anchor_offset) {
  int offset = 0;
  for (int i = 0; i < bbox_num; ++i) {
    float no_face = h2f_tool->convert(data[offset + 0]);
    float face = h2f_tool->convert(data[offset + 1]);
    float score = SlimSoftMax(face, no_face);
    if (score > g_fd_conf_thresh)  // filter out the face under g_conf_thresh
    {
      Face face;
      // decode bbox
      Anchor &anchor = anchors_[anchor_offset + i];
      float center_x = g_fd_prior_variance[0] * (h2f_tool->convert(data[offset + 2])) * anchor.s_kx + anchor.cx;
      float center_y = g_fd_prior_variance[0] * (h2f_tool->convert(data[offset + 3])) * anchor.s_ky + anchor.cy;
      float bbox_width = expf(g_fd_prior_variance[1] * (h2f_tool->convert(data[offset + 4]))) * anchor.s_kx;
      float bbox_height = expf(g_fd_prior_variance[1] * (h2f_tool->convert(data[offset + 5]))) * anchor.s_ky;
      face.xmin = (center_x - bbox_width / 2) * g_image_w;
      face.ymin = (center_y - bbox_height / 2) * g_image_h;
      face.xmax = (center_x + bbox_width / 2) * g_image_w;
      face.ymax = (center_y + bbox_height / 2) * g_image_h;
      face.area = (bbox_width * g_image_w + 1) * (bbox_height * g_image_h + 1);
      face.score = score;
      filtered_face_.push_back(std::move(face));
    }
    offset += 16;
  }
}
float FaceDetection::Overlap(const Face &bbox1, const Face &bbox2) {
  if (bbox1.xmax < bbox2.xmin || bbox1.ymax < bbox2.ymin || bbox2.xmax < bbox1.xmin || bbox2.ymax < bbox1.ymin) {
    return 0.0;
  } else {
    const float inter_xmin = std::max(bbox1.xmin, bbox2.xmin);
    const float inter_ymin = std::max(bbox1.ymin, bbox2.ymin);
    const float inter_xmax = std::min(bbox1.xmax, bbox2.xmax);
    const float inter_ymax = std::min(bbox1.ymax, bbox2.ymax);
    const float inter_width = inter_xmax - inter_xmin + 1;
    const float inter_height = inter_ymax - inter_ymin + 1;
    const float inter_size = inter_width * inter_height;
    return inter_size / (bbox1.area + bbox2.area - inter_size);
  }
}
void FaceDetection::FaceNMS() {
  std::stable_sort(filtered_face_.rbegin(), filtered_face_.rend(), comp);  // sort by score
  int top_count = 0;
  for (int i = 0; i < filtered_face_.size(); i++) {
    if (top_count == g_fd_keep_top_k) break;
    if (filtered_face_[i].score < g_fd_conf_thresh) continue;
    for (int j = i + 1; j < filtered_face_.size(); j++) {
      if (filtered_face_[j].score < g_fd_conf_thresh) continue;
      float overlap = Overlap(filtered_face_[i], filtered_face_[j]);
      if (overlap > g_fd_nms_thresh)  // erase the bbox that has bigger overlap
      {
        filtered_face_[j].score = 0;
      }
    }
    top_count++;
    nms_out_face_.push_back(filtered_face_[i]);
  }
}
Face FaceDetection::ExtractFace(uint16_t *data[], const int layer_num) {
  // make sure there is no face data in filtered_face_
  filtered_face_.clear();

  if (layer_num != g_fd_out_layer_num) {
    std::cout << "Error ,the layer_num nor g_out_layer_num is not right!!!" << std::endl;
  }

  // store the face whose score is above g_conf_thresh in filtered_face_
  int anchor_offset = 0;
#ifdef USE_G3_MODEL  // G3 model
  for (int i = 0; i < layer_num; ++i) {
    int bbox_num = g_fd_output_size[i] * g_fd_output_size[i];
    this->FilterFace(data[i], bbox_num, anchor_offset);
    anchor_offset += bbox_num;
  }
#else  // P7 model
  for (int i = 0; i < 2; ++i) {
    int bbox_num = g_fd_output_size[0] * g_fd_output_size[1];
    this->FilterFace(data[0] + i * 6, bbox_num, anchor_offset);
    anchor_offset += bbox_num;
  }
#endif

  // do nms on filtered_face_ after FilterFace and store the output to nms_out_face_
  this->FaceNMS();
  // may do something to choose the output face
  Face face_out = {0, 0, 0, 0, 0};
  if (nms_out_face_.size() > 0) {
    face_out = nms_out_face_[0];
  }

  filtered_face_.clear();
  nms_out_face_.clear();
  return face_out;
}
