#ifndef TASK_EDGE_LINE_DECODER_HPP
#define TASK_EDGE_LINE_DECODER_HPP

#include <assert.h>
#include <iostream>
#include <string>
#include <vector>
#include <deque>

#include "../../type_def.hpp"
#include "task_utils.hpp"
#include "../../networks/perception_config_multicam.hpp"

struct PointAttr {
  bool visible = true;         // true: visible, false: not visible
  bool hanging = false;           // true: hang in the air, false: not hang in the air
  bool grass_covered = false;  // true: covered by grass, false: not covered by grass
};

namespace lld {
  struct Point2f {
    float x_;
    float y_;
    float conf_;
    float line_embedding_;
    int category_;
    float orientation_;
    PointAttr attr_;

    Point2f() {
      x_ = -1;
      y_ = -1;
      conf_ = -1;
      line_embedding_ = -1;
      category_ = -1;
      orientation_ = -1;
    }

    Point2f (float x, float y) {
      x_ = x;
      y_ = y;
      conf_ = -1;
      line_embedding_ = -1;
      category_ = -1;
      orientation_ = -1;
    }

    Point2f (int x, int y) {
      x_ = x;
      y_ = y;
      conf_ = -1;
      line_embedding_ = -1;
      category_ = -1;
      orientation_ = -1;
    }

    Point2f (float x, float y, float conf, float line_embedding, int category, float orientation, PointAttr attr) {
      x_ = x;
      y_ = y;
      conf_ = conf;
      line_embedding_ = line_embedding;
      category_ = category;
      orientation_ = orientation;
      attr_ = attr;
    }

    float &x() {return x_;}
    float &y() {return y_;}
    float &conf() {return conf_;}
    float &line_embedding() {return line_embedding_;}
    int &category() {return category_;}
    float &orientation() {return orientation_;}
    PointAttr &attr() {return attr_;}
  };

  struct Point2fWithVariance {
    Point2f pt_;
    float var_;

    Point2fWithVariance() : var_(-1){}
    Point2fWithVariance(Point2f pt, float var) : pt_(pt), var_(var){}
    Point2f &pt() {return pt_;}
    float &var() {return var_;}

    bool operator==(const Point2fWithVariance &pt) const {
      return (pt_.x_ == pt.pt_.x_ && pt_.y_ == pt.pt_.y_);
    }
  };

  typedef std::vector<Point2fWithVariance> PointList2f;
}

struct AdHeatMapPoints2D {
  int x_;
  int y_;
  float px_;
  float py_;
  float w_;
  float h_;
  float heatmap_value_;
  int index_;
  float embedding_;  // point embedding
  float line_embedding_;  // line embedding
  int category_;
  float orientation_;
  PointAttr attr_;

  explicit AdHeatMapPoints2D(int x, int y, float px, float py, float w, float h, float v, int index,
      float embedding = 0.0f, float line_embedding = 0, int category = -1, float orientation = 0, PointAttr attr = PointAttr())
      : x_(x), y_(y), px_(px), py_(py), w_(w), h_(h), heatmap_value_(v), index_(index), embedding_(embedding),
        line_embedding_(line_embedding), category_(category), orientation_(orientation), attr_(attr) {}
};

/// @brief lld predict channels index
static constexpr int lld_pred_channels_index[5] = {0, 1, 2, 3, 4};

static constexpr int kLLDClsStartIndex = 5;
static constexpr int kLLDClsEndIndex = 14;

static constexpr int kLLDOrient1Index = 14;
static constexpr int kLLDOrient2Index = 15;

static constexpr int kLLDVisibleIndex = 16;
static constexpr int kLLDHangingIndex = 17;
static constexpr int kLLDGrassCoveredIndex = 18;

/// @brief AP LLD max lanes numbers
static constexpr int kAPMaxLanesNums = 2;

/// @brief scale from (914, 474) to (457, 237)
static constexpr float kAPLLDPointScale = 1;  // 0.5;

/// @brief anchor projection back to py point - top size value
static constexpr int kCropSizeTop = 0;  //30;

/// @brief anchor projection back to px point - left size value
static constexpr int kCropSizeLeft = 0;

/// @brief anchor projection back to px point - left pad value
static constexpr int kMapPadLeft = 0;

/// @brief anchor projection back to px point - top pad value
static constexpr int kMapPadTop = 0;

/// @brief near area range confidence threshhold -- near area refer to py [kFarRangeStart, kFarRangeEnd]
static constexpr float kNearConfThresh  = 0.2;

/// @brief far area range start
static constexpr int kFarRangeStart = 0;

/// @brief far area range end
static constexpr int kFarRangeEnd = 10;

/// @brief gap threshhold between clusters
static constexpr float kClusterGapThresh = 0.8;  // 0.5;

/// @brief max nums for lanes
static constexpr int kMaxPredLane = 20;

/// @brief max nums for points
static constexpr float kMaxPointListLen = 51200;

/// @brief max nums for piecewise lanes
static constexpr float kMaxPiecewiseLineLen = 20;

/// @brief distance threshold for piecewise lanes
static constexpr float kDistanceThreshold = 16;


/// @brief Class TaskEdgeLld is responsible to detect lld result
class TaskEdgeLld {
 public:
  explicit TaskEdgeLld(const std::string &mode, const std::string &camera_name);
  ~TaskEdgeLld();

  /// @brief decode LLD and Arrow Function
  ///
  /// @param infer_config
  /// @param camera_name
  /// @param label, network output
  /// @param lanes_output, save lines result for dds
  /// @param ap_rsm_obj_list, save arrow objs for dds
  /// @return None
  void DecodeAPLLD(const float *data, std::vector<EdgeLine> &lines);

 private:
  /// @brief decode LLD Lanes Function
  ///
  /// @param label, network output
  /// @param res_lanes, save lines list for dds
  /// @return None
  void DecodeAPLanes(const float *label, std::vector<lld::PointList2f> &res_lanes);

  /// @brief Cluster LLD Lanes Points Function
  ///
  /// @return None
  void ClusterLinePoints();

  /// @brief Split line points by category
  ///
  /// @return None
  void SplitLinePointsByCls(std::vector<lld::PointList2f> &pred_lanes);

  /// @brief Slim line
  ///
  /// @return slim points
  std::vector<lld::Point2fWithVariance> GetSlimLinePoint(const lld::PointList2f &line, const int fixed,
                                                         const bool by_order_y);

  /// @brief Slim line
  ///
  /// @return slim line
  lld::PointList2f SlimLine(const lld::PointList2f &line, const float min, const float max, const int pixel_step,
                            const bool by_order_y);

  /// @brief Slim lines
  ///
  /// @return None
  void SlimLines(std::vector<lld::PointList2f> &pred_lanes);

  /// @brief Smooth lines
  ///
  /// @return None
  void FilterLines(std::vector<lld::PointList2f> &res_lanes);

  /// @brief Smooth lines
  ///
  /// @return None
  void ReverseLines(std::vector<lld::PointList2f> &res_lanes);

  /// @brief Split piecewise lines
  /// @param piecewise_lines
  /// @param split_dist
  void SplitPiecewiseLines(std::vector<lld::PointList2f> &piecewise_lines, const float split_dist=12);

  /// @brief Zhang-Suen thining algorithm
  /// \param line
  void ZhangSuenThiningPoints(lld::PointList2f &line);

  /// @brief Arrange Point To LLD Lanes Function
  ///
  /// @param selected_points, each lines for selected points list
  /// @param res_points, each lines for result points
  /// @return None
  void ArrangePoints2Line(lld::PointList2f &selected_points, lld::PointList2f &res_points);

  /// @brief Serialize LLD Lanes Function
  ///
  /// @param pred_lanes, each cluster result lines
  /// @param new_lanes,  each Serialize result lines
  /// @return None
  void SerializeSingleLine(lld::PointList2f &pred_lanes, lld::PointList2f &new_lanes);

  /// @brief  Filter out the overlapping points and choose the one with the highest confidence
  /// @param selected_points, each points before filter
  /// @param selected_points_out_,  each points after  overlapping filter by confidence
  /// @return None
void PickHighConfPoints(lld::PointList2f &selected_points, lld::PointList2f & selected_points_out_);
  /// @brief Extend LLD Lanes End Points Function
  ///
  /// @param selected_points, each lines for selected points
  /// @param pred_lanes,  each Extend LLD Lanes
  /// @return None

  void ExtendEndPoints(lld::PointList2f &selected_points, lld::PointList2f &pred_lanes);

  /// @brief Connect LLD Piecewise Lanes
  ///
  /// @param piecewise_lines, piecewise lines
  /// @param res_lanes, connect result lines
  /// @return None
  void ConnectPiecewiseLines(std::vector<lld::PointList2f> &piecewise_lines, lld::PointList2f &res_lanes);

  /// @brief Sort Conf Indices
  ///
  /// @param confs, confidence value
  /// @param num_boxes, bbox nums
  /// @param idx, sorted index
  /// @return None
  void SortIndices(const float *confs, const int num_boxes, std::vector<int> &idx);

  // common functions
  float Sigmoid(float x);

  void ShowLine(lld::PointList2f &line, const std::string &title = "show");
  void ShowLines(const std::vector<lld::PointList2f> &lines, const std::string &title = "show");

  float origin_img_width_;
  float origin_img_height_;

  /// @brief network lld branch input image width / 2
  float image_width_;

  /// @brief network lld branch input image height / 2
  float image_height_;

  /// @brief network lld branch featuremap width
  int featuremap_width_;

  /// @brief network lld branch featuremap height
  int featuremap_height_;

  /// @brief network lld branch featuremap size
  int featuremap_size_;

  /// @brief decode anchor scale
  int anchor_scale_;

  /// @brief far area range end
  int far_range_end_;

  /// @brief far area range confidence threshhold -- far area refer to py [kFarRangeEnd, image_height]
  float far_confidence_threshold_;

  /// @brief remove short line threshold in cluster
  int long_line_threshold_;

  /// @brief save predict lanes
  std::vector<lld::PointList2f> pred_lanes_;

  /// @brief save dds result lanes
  std::vector<lld::PointList2f> lanes_list_;

  /// @brief output dds lanes result
  // lld::Lanes lanes_result_;

  /// @brief cluster center means for calculate center nums in ClusterLinePoints
  std::vector<float> embedding_means_;

  /// @brief the number of points in each cluster center in ClusterLinePoints
  std::vector<int> point_numbers_;

  /// @brief piecewise lines in SerializeSingleLine
  std::vector<lld::PointList2f> piecewise_lines_;

  /// @brief coordinate x value in ArrangePoints2Line
  std::vector<float> x_value_;

  /// @brief coordinate y value in ArrangePoints2Line
  std::vector<float> y_value_;

  /// @brief confidence value in ArrangePoints2Line
  std::vector<float> confidence_value_;

  std::vector<float> line_embedding_value_;
  std::vector<int> category_value_;
  std::vector<float> orientation_value_;
  std::vector<PointAttr> attr_value_;

  /// @brief sort index in ArrangePoints2Line
  std::vector<int> idx_;

  /// @brief long lines in ConnectPiecewiseLines
  std::vector<lld::PointList2f> long_lines_;

  /// @brief final lines in ConnectPiecewiseLines
  std::vector<lld::PointList2f> final_lines_;

  /// @brief other lines in ConnectPiecewiseLines
  std::vector<lld::PointList2f> other_lines_;

  /// @brief current end point in ConnectPiecewiseLines
  std::vector<lld::Point2fWithVariance> current_end_point_;

  /// @brief other end point in ConnectPiecewiseLines
  std::vector<lld::Point2fWithVariance> other_end_point_;

  std::deque<AdHeatMapPoints2D> idx_list_;

  const std::string mode_;
  const std::string camera_name_;
  InferenceConfig *infer_config;
};
/// @brief sort point by x dim  from min value to max value
bool sort_point_x_dim(lld::Point2fWithVariance point_a,  lld::Point2fWithVariance point_b);
/// @brief sort point by y dim  from min value to max value
bool sort_point_y_dim(lld::Point2fWithVariance point_a,  lld::Point2fWithVariance point_b);
#endif  // TASK_EDGE_LINE_DECODER_HPP
