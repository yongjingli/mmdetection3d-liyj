#ifndef FACE_DETECTION_H_
#define FACE_DETECTION_H_
#include <memory>
#include <vector>
#include "half_convert.h"
struct Face {
  float score, xmin, ymin, xmax, ymax, area;
};
struct Anchor {
  float cx, cy, s_kx, s_ky;
  Anchor(float cx1, float cy1, float s_kx1, float s_ky1) : cx(cx1), cy(cy1), s_kx(s_kx1), s_ky(s_ky1) {}
};

class FaceDetection {
 private:
  inline float SlimSoftMax(const float score1, const float score2);
  void FilterFace(const uint16_t *data, const int bbox_num, const int anchor_offset);
  float Overlap(const Face &bbox1, const Face &bbox2);
  void GenPriorbox();
  void FaceNMS();                    // do NMS on filtered_face_ after FilterFace
  std::vector<Face> filtered_face_;  // store the face that filtered by g_conf_thresh
  std::vector<Anchor> anchors_;      // store the achors that created by GenPriorbox function
  std::vector<Face> nms_out_face_;   // store the output face after NMS function
  std::unique_ptr<half2float> h2f_tool;

 public:
  FaceDetection();
  ~FaceDetection(){};
  Face ExtractFace(uint16_t *data[], const int layer_num);
};

#endif  // FACE_DETECTION_H_