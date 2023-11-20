#include <algorithm>
#include <cmath>
#include <numeric>

#include "task_edge_line_decoder.hpp"

TaskEdgeLld::TaskEdgeLld(const std::string &mode, const std::string &camera_name) : mode_(mode),
                                                                                    camera_name_(camera_name) {

  infer_config = PerceptionConfig::GetInferenceConfig(mode_, camera_name);

  const auto &camera_config = PerceptionConfig::GetCameraConfig(mode_, camera_name);

  origin_img_width_ = camera_config->image_width;
  origin_img_height_ = camera_config->image_height;

  // reused infer config in ap mode
  image_height_ = infer_config->edge_line_input_height;
  image_width_ = infer_config->edge_line_input_width;
  far_range_end_ = infer_config->edge_line_far_range_end;
  far_confidence_threshold_ = infer_config->edge_line_far_conf_thr;
  long_line_threshold_ = infer_config->edge_line_long_line_thr;

  featuremap_width_ = infer_config->edge_line_output_width;
  featuremap_height_ = infer_config->edge_line_output_height;
  featuremap_size_ = featuremap_width_ * featuremap_height_;
  anchor_scale_ = infer_config->edge_line_output_scale;

  pred_lanes_.resize(kMaxPredLane);
  // lanes_result_.pred_lanes().resize(kAPMaxLanesNums);
  piecewise_lines_.reserve(kMaxPiecewiseLineLen);
}

TaskEdgeLld::~TaskEdgeLld() = default;

inline float TaskEdgeLld::Sigmoid(float x) { return (1 / (1 + expf(-x))); }

inline float ConvertToMEBOW(const float orientation) {
  if (180 >= orientation && orientation >= 0)
    return 90 + (180 - orientation);
  else if (orientation >= -90)
    return 270 + fabs(orientation);
  else if (orientation >= -180)
    return fabs(orientation) - 90;
  else
    return -1;
}

float CalculatePointsOrientation(const lld::Point2fWithVariance &pre_point, const lld::Point2fWithVariance &curr_point) {
  // 得到pre_point指向cur_point的方向
  // 转为以pre_point为原点的坐标系
  float x1 = pre_point.pt_.x_;
  float y1 = pre_point.pt_.y_;
  float x2 = curr_point.pt_.x_;
  float y2 = curr_point.pt_.y_;
  float x = x2 - x1;
  float y = y2 - y1;

  // 转为与人体朝向的坐标定义类似，以正前方的指向为0，然后逆时针得到360的朝向
  // 记住图像的坐标系为y向下,x向右
  float orient = -1;
  if (x != 0) {
    float angle = fabs(atan(y / x)) / PI * 180;
    // 判断指向所在的象限
    // 在3、4象限
    if (y >= 0) {
      // 在3象限
      if (x < 0) {
        orient = 90 + angle;
      } else {
        // 在4象限
        orient = 180 + (90 - angle);
      }
    } else { // 在1、2象限
      // 在1象限
      if (x > 0) {
        orient = 270 + angle;
      } else { // 在2象限
        orient = 90 - angle;
      }
    }
  } else {
    // 当x为0的时候
    if (y >= 0) {
      if (y == 0) {
        orient = -1;
      } else {
        orient = 180;
      }
    } else {
      orient = 0;
    }
  }
  return orient;
}

void TaskEdgeLld::DecodeAPLLD(const float *data, std::vector<EdgeLine> &lines) {
  // decode LLD Lanes
  lanes_list_.clear();
  DecodeAPLanes(data, lanes_list_);

  lines.resize(lanes_list_.size());

  float scale_x, scale_y, offset_x, offset_y;
  GetScaleOffset(origin_img_width_, origin_img_height_, image_width_, image_height_, scale_x, scale_y, offset_x,
                 offset_y, false);

  std::vector<float> line_embeddings(lanes_list_.size());

  for (int i = 0; i < lanes_list_.size(); i++) {
    auto &line = lines[i];
    line.id = i;
    line.points.resize(lanes_list_[i].size());

    float mean_line_embedding = 0;
    int category_num = kLLDClsEndIndex - kLLDClsStartIndex;
    std::vector<int> category_count(category_num, 0);
    for (int p = 0; p < lanes_list_[i].size(); p++) {
      // rescale to origin image size
      line.points[p].x = lanes_list_[i][p].pt().x() * scale_x + offset_x;
      line.points[p].y = lanes_list_[i][p].pt().y() * scale_y + offset_y;
      line.points[p].z = -1;
      line.points[p].conf = lanes_list_[i][p].pt().conf();
      line.points[p].category = lanes_list_[i][p].pt().category();
      line.points[p].attribute.clear();
      line.points[p].attribute.push_back(lanes_list_[i][p].pt().attr_.visible);
      line.points[p].attribute.push_back(lanes_list_[i][p].pt().attr_.hanging);
      line.points[p].attribute.push_back(lanes_list_[i][p].pt().attr_.grass_covered);

      category_count[lanes_list_[i][p].pt().category()]++;
      mean_line_embedding += lanes_list_[i][p].pt().line_embedding();
    }
    mean_line_embedding /= lanes_list_[i].size();
    line_embeddings[i] = mean_line_embedding;

    // max count pt category as line category
    line.category = std::distance(category_count.begin(), std::max_element(category_count.begin(), category_count.end()));
    line.category_name = GetEdgeLineCategories()[line.category];
  }

  // check whether 2 lines are same line by line embedding, same line use same id
  float line_embedding_thr = 0.5;
  for (int i = 0; i < line_embeddings.size(); i++) {
    for (int j = i + 1; j < line_embeddings.size(); j++) {
      if (std::fabs(line_embeddings[i] - line_embeddings[j]) < line_embedding_thr) {
        lines[j].id = lines[i].id;
      }
    }
  }
}

void TaskEdgeLld::DecodeAPLanes(const float *label, std::vector<lld::PointList2f> &res_lanes) {
  const bool &lld_confidence_filter = infer_config->edge_line_confidence_filter;
  const int &lld_candidate_size = infer_config->edge_line_candidate_size;
  const int offset = infer_config->edge_line_confidence_filter ? 1 : 0;
  const int output_size_of_pt = infer_config->edge_line_candidate_pt_size + offset;

  const int channel_index_exist = lld_pred_channels_index[0];
  const int channel_index_px = lld_pred_channels_index[1];
  const int channel_index_py = lld_pred_channels_index[2];
  const int channel_index_embed = lld_pred_channels_index[3];
  const int channel_index_line_embed = lld_pred_channels_index[4];

  DCHECK(featuremap_size_ > 0);
  DCHECK(featuremap_width_ > 0);
  DCHECK(featuremap_height_ > 0);
  DCHECK(anchor_scale_ > 0);

  const int hm_size = lld_confidence_filter ? lld_candidate_size : featuremap_size_;
  float px, py, conf, embedding, line_embedding, orientation;
  int locate_id = 0, category;
  std::vector<float> category_confs;
  const int *hm_id_ptr = reinterpret_cast<const int *>(label);
  idx_list_.clear();
  for (int i = 0; i < hm_size; i++) {
    if (lld_confidence_filter) {
      locate_id = hm_id_ptr[i * output_size_of_pt];
      if (locate_id < 0) {
        break;
      }
    }

    int id = lld_confidence_filter ? locate_id : i;
    int row_i = static_cast<int>(id / featuremap_width_);
    int col_i = id - row_i * featuremap_width_;
    int x_left_offset = anchor_scale_ * col_i + kCropSizeLeft * anchor_scale_ - kMapPadLeft;
    int y_top_offset = anchor_scale_ * (row_i + kCropSizeTop) - kMapPadTop;
    conf = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + channel_index_exist + 1]
                                         : label[channel_index_exist * featuremap_size_ + i]);

    bool valid_point = false;
    if (row_i >= kFarRangeStart && row_i < far_range_end_) {
      if (conf > far_confidence_threshold_) {
        valid_point = true;
      }
    } else {
      if (conf > kNearConfThresh) {
        valid_point = true;
      }
    }

    if (valid_point) {
      px = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + channel_index_px + 1]
                                         : label[channel_index_px * featuremap_size_ + i]) *
          (anchor_scale_ - 1) +
          x_left_offset;
      py = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + channel_index_py + 1]
                                         : label[channel_index_py * featuremap_size_ + i]) *
          (anchor_scale_ - 1) +
          y_top_offset;
      embedding = lld_confidence_filter ? label[i * output_size_of_pt + channel_index_embed + 1]
                                        : label[channel_index_embed * featuremap_size_ + i];
      line_embedding = lld_confidence_filter ? label[i * output_size_of_pt + channel_index_line_embed + 1]
                                             : label[channel_index_line_embed * featuremap_size_ + i];
      // note: not enable confidence filter
      GetNetworkAttrMaxIdx(label, col_i, row_i, kLLDClsStartIndex, kLLDClsEndIndex, featuremap_width_,
                           featuremap_height_, category);
      float ori1 = lld_confidence_filter ? label[i * output_size_of_pt + kLLDOrient1Index + 1]
                                         : label[kLLDOrient1Index * featuremap_size_ + i];
      float ori2 = lld_confidence_filter ? label[i * output_size_of_pt + kLLDOrient2Index + 1]
                                         : label[kLLDOrient2Index * featuremap_size_ + i];
      ori1 = Sigmoid(ori1) * 2 - 1;
      ori2 = Sigmoid(ori2) * 2 - 1;
      orientation = atan2(ori1, ori2) / PI * 180;
      orientation = ConvertToMEBOW(orientation);

      float visible = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + kLLDVisibleIndex + 1]
                                                    : label[kLLDVisibleIndex * featuremap_size_ + i]);
      float hanging = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + kLLDHangingIndex + 1]
                                                    : label[kLLDHangingIndex * featuremap_size_ + i]);
      float grass_covered = Sigmoid(lld_confidence_filter ? label[i * output_size_of_pt + kLLDGrassCoveredIndex + 1]
                                                          : label[kLLDGrassCoveredIndex * featuremap_size_ + i]);
      PointAttr attr;
      attr.visible = visible > 0.5;
      attr.hanging = hanging > 0.5;
      attr.grass_covered = grass_covered > 0.5;

      idx_list_.emplace_back(col_i, row_i, px, py, 0.0f, 0.0f, conf, i, embedding, line_embedding, category, orientation, attr);
    }
  }

  std::stable_sort(idx_list_.begin(), idx_list_.end(),
                   [](const AdHeatMapPoints2D &lhs, const AdHeatMapPoints2D &rhs) { return lhs.y_ > rhs.y_; });

  ClusterLinePoints();

#if 0  // only keep the longest line
  res_lanes.resize(pred_lanes_.size());
  for (int i = 0; i < pred_lanes_.size(); i++) {
    if (pred_lanes_[i].size() > 0) {
      SerializeSingleLine(pred_lanes_[i], res_lanes[i]);
    }
  }
#else  // keep all piecewise lines
  res_lanes.clear();
  for (int i = 0; i < pred_lanes_.size(); i++) {
    if (pred_lanes_[i].size() > 0) {
      lld::PointList2f single_line;
      SerializeSingleLine(pred_lanes_[i], single_line);
      // keep all long piecewise lines
      if (!single_line.empty()) {
        res_lanes.push_back(final_lines_[0]);
        for (int j = 1; j < final_lines_.size(); j++) {
          if (final_lines_[j].size() > 0) {
            res_lanes.push_back(final_lines_[j]);
          }
        }
      }
    }
  }
#endif

  ReverseLines(res_lanes);

  // filter_lines, smooth line
  FilterLines(res_lanes);
}

void TaskEdgeLld::FilterLines(std::vector<lld::PointList2f> &res_lanes) {
  const int pixel_err_thr = 20;
  const int min_pt_num = 10;
  const int kernel_half_size = 2;

  std::vector<lld::PointList2f> new_lines;
  for (auto &line : res_lanes) {
    if (line.size() <= min_pt_num) continue;

    lld::PointList2f smooth_line;
    // not smooth begin half kernel
    for (int i = 0; i < kernel_half_size; i++) {
      smooth_line.push_back(line[i]);
    }

    // smooth
    for (int i = kernel_half_size; i < line.size() - kernel_half_size; i++) {
      float sum_x = 0;
      float sum_y = 0;
      for (int j = i - kernel_half_size; j <= i + kernel_half_size; j++) {
        sum_x += line[j].pt().x();
        sum_y += line[j].pt().y();
      }
      float mean_x = sum_x / (kernel_half_size * 2 + 1);
      float mean_y = sum_y / (kernel_half_size * 2 + 1);

      // discard this point if error too big
      float err = std::fabs(line[i].pt().x() - mean_x) + std::fabs(line[i].pt().y() - mean_y);
      if (err < pixel_err_thr) {
        lld::Point2fWithVariance pt = line[i];
        pt.pt().x() = mean_x;
        pt.pt().y() = mean_y;
        smooth_line.push_back(pt);
      }
    }

    // not smooth end half kernel
    for (int i = line.size() - kernel_half_size; i < line.size(); i++) {
      smooth_line.push_back(line[i]);
    }

    if (smooth_line.size() > 1) {
      new_lines.push_back(smooth_line);
    }
  }

  res_lanes = new_lines;
}

void TaskEdgeLld::ClusterLinePoints() {
  embedding_means_.clear();
  point_numbers_.clear();

  // alloc pred_lanes_ size
  pred_lanes_.resize(kMaxPredLane);
  for (int i = 0; i < kMaxPredLane; i++) {
    pred_lanes_[i].clear();
  }

  if (idx_list_.size() > 0) {
    pred_lanes_[0].push_back(lld::Point2fWithVariance({{(float) idx_list_[0].x_, (float) idx_list_[0].y_,
                                                        idx_list_[0].heatmap_value_, idx_list_[0].line_embedding_,
                                                        idx_list_[0].category_, idx_list_[0].orientation_,
                                                        idx_list_[0].attr_}, 0}));
    embedding_means_.emplace_back(idx_list_[0].embedding_);
    point_numbers_.emplace_back(1);
  }

  for (int i = 1; i < idx_list_.size(); i++) {
    int lanes_index = -1;
    float min_dist = 10000;
    for (int j = 0; j < embedding_means_.size(); j++) {
      float distance = std::abs(idx_list_[i].embedding_ - embedding_means_[j]);
      if (distance < kClusterGapThresh && distance < min_dist) {
        lanes_index = j;
        min_dist = distance;
      }
    }

    if (lanes_index == -1) {
      if (embedding_means_.size() >= kMaxPredLane) {
        continue;
      }
      lanes_index = embedding_means_.size();
    }

    // dds limit 128 points
    if (pred_lanes_[lanes_index].size() < kMaxPointListLen) {
      pred_lanes_[lanes_index].push_back(lld::Point2fWithVariance({{(float) idx_list_[i].x_, (float) idx_list_[i].y_,
                                                                    idx_list_[i].heatmap_value_,
                                                                    idx_list_[i].line_embedding_,
                                                                    idx_list_[i].category_, idx_list_[i].orientation_,
                                                                    idx_list_[i].attr_},
                                                                   0}));
    }

    if (lanes_index >= embedding_means_.size()) {
      // push a new cluster center
      embedding_means_.emplace_back(idx_list_[i].embedding_);
      point_numbers_.emplace_back(1);
    } else {
      // calculate the new cluster means of the current cluster center
      embedding_means_[lanes_index] =
          (embedding_means_[lanes_index] * point_numbers_[lanes_index] + idx_list_[i].embedding_) /
          (point_numbers_[lanes_index] + 1);
      point_numbers_[lanes_index] += 1;
    }
  }

  pred_lanes_.resize(embedding_means_.size());

  // split_line_points_by_cls
  SplitLinePointsByCls(pred_lanes_);

  // get_slim_lines
  SlimLines(pred_lanes_);

  /*for (int l = 0; l < pred_lanes_.size(); l++) {
    std::sort(pred_lanes_[l].begin(), pred_lanes_[l].end(), [](lld::Point2fWithVariance a, lld::Point2fWithVariance b) {
      return a.pt().y() < b.pt().y();
    });
  }*/

  cv::Mat show = cv::Mat::ones(152, 240, CV_8UC3) * 255;
  for (int l = 0; l < 1; l++) {
    for (int p = 0; p < pred_lanes_[l].size(); p++) {
      show.at<cv::Vec3b>(pred_lanes_[l][p].pt().y(), pred_lanes_[l][p].pt().x()) = cv::Vec3b(l * 10, l * 10, l * 10);
    }
  }
  cv::namedWindow("show", 0);
  cv::imshow("show", show);
  cv::waitKey(0);

  for (int i = 0; i < pred_lanes_.size(); i++) {
    // Remove short lines
    if (pred_lanes_[i].size() < long_line_threshold_) {
      pred_lanes_[i].clear();
      continue;
    }

    // Remove far range lines
    bool isfarline = true;
    for (auto &point : pred_lanes_[i]) {
      if (point.pt().y() >= kFarRangeEnd) {
        isfarline = false;
        break;
      }
    }
    if (isfarline) {
      pred_lanes_[i].clear();
    }
  }
}

void TaskEdgeLld::SplitLinePointsByCls(std::vector<lld::PointList2f> &pred_lanes) {
  std::vector<lld::PointList2f> new_lines;
  for (auto &line : pred_lanes) {
    std::unordered_map<int, lld::PointList2f> split_category_lines;
    for (auto &pt : line) {
      split_category_lines[pt.pt().category()].push_back(pt);
    }
    for (auto &kv : split_category_lines) {
      new_lines.push_back(kv.second);
    }
  }
  pred_lanes = new_lines;
}

// by_order_y: 线是否y方向更长，是的话则fixed表示y的坐标，y==fixed的点根据x排序找中值点；
// 反之则线的x方向更长，fixed表示x坐标，x==fixed的点根据y排序找中值点
std::vector<lld::Point2fWithVariance> TaskEdgeLld::GetSlimLinePoint(const lld::PointList2f &line, const int fixed,
                                                       const bool by_order_y) {
  std::vector<lld::Point2fWithVariance> pts;
  if (!by_order_y) {
    for (auto &pt : line) {
      if ((int)pt.pt_.x_ == fixed) {
        pts.push_back(pt);
      }
    }
    std::sort(pts.begin(), pts.end(), sort_point_y_dim);
  } else {
    for (auto &pt : line) {
      if ((int)pt.pt_.y_ == fixed) {
        pts.push_back(pt);
      }
    }
    std::sort(pts.begin(), pts.end(), sort_point_x_dim);
  }

  if (pts.size() <= 1) {
    return pts;
  }

  std::vector<lld::Point2fWithVariance> mid_pts;
  int begin_id = 0;
  for (int i = 1; i < pts.size(); i++) {
    int dist;
    if (!by_order_y) {
      dist = abs((int)pts[i].pt().y() - (int)pts[i-1].pt().y());
    } else {
      dist = abs((int)pts[i].pt().x() - (int)pts[i-1].pt().x());
    }
    if (dist > 1) {
      int mid_id = (i + begin_id) / 2;
      mid_pts.push_back(pts[mid_id]);
      begin_id = i;
    }

    if (i == pts.size() - 1) {
      int mid_id = (i + begin_id) / 2;
      mid_pts.push_back(pts[mid_id]);
    }
  }

  return mid_pts;
}

lld::PointList2f TaskEdgeLld::SlimLine(const lld::PointList2f &line, const float min, const float max,
                                       const int pixel_step, const bool by_order_y) {
  lld::PointList2f slim_line;
  for (int i = (int)min; i <= (int)max; i += pixel_step) {
    std::vector<lld::Point2fWithVariance> keypoints = GetSlimLinePoint(line, i, by_order_y);
    slim_line.insert(slim_line.end(), keypoints.begin(), keypoints.end());
  }
  return slim_line;
}

void TaskEdgeLld::SlimLines(std::vector<lld::PointList2f> &pred_lanes) {
  std::vector<lld::PointList2f> slim_lines;
  for (auto &line : pred_lanes) {
    if (line.size() == 0) continue;

    // find min and max
    float xmin = line[0].pt().x();
    float ymin = line[0].pt().y();
    float xmax = xmin;
    float ymax = ymin;
    for (auto &pt : line) {
      float x = pt.pt().x();
      float y = pt.pt().y();
      xmin = x < xmin ? x : xmin;
      xmax = x > xmax ? x : xmax;
      ymin = y < ymin ? y : ymin;
      ymax = y > ymax ? y : ymax;
    }

    int pixel_step = 1;
    float x_len = xmax - xmin;
    float y_len = ymax - ymin;
    float ratio_len = std::max(x_len, y_len) / (std::min(x_len, y_len) + 1e-8);

    lld::PointList2f slim_line;
    if (ratio_len > 2) {
      if (x_len > y_len) {
        slim_line = SlimLine(line, xmin, xmax, pixel_step, false);
      } else {
        slim_line = SlimLine(line, ymin, ymax, pixel_step, true);
      }
    } else {
      lld::PointList2f slim_line_x = SlimLine(line, xmin, xmax, pixel_step, false);
      lld::PointList2f slim_line_y = SlimLine(line, ymin, ymax, pixel_step, true);
      slim_line = slim_line_x;
      slim_line.insert(slim_line.end(), slim_line_y.begin(), slim_line_y.end());
    }

    slim_lines.push_back(slim_line);
  }

  pred_lanes = slim_lines;
}

void TaskEdgeLld::ConnectPiecewiseLines(std::vector<lld::PointList2f> &piecewise_lines, lld::PointList2f &res_lanes) {
  long_lines_ = piecewise_lines;
  final_lines_.clear();
  while (long_lines_.size() > 1) {
    // Find the end point of the first line
    lld::PointList2f current_line = long_lines_[0];
    current_end_point_.clear();
    current_end_point_.push_back(current_line[0]);
    if (current_line.size() > 1) {
      current_end_point_.push_back(current_line[current_line.size() - 1]);
    }
    // Find the end point of the other line
    other_lines_.clear();
    for (int i = 1; i < long_lines_.size(); i++) {
      other_lines_.push_back(long_lines_[i]);
    }
    other_end_point_.clear();
    for (auto &other_line : other_lines_) {
      other_end_point_.push_back(other_line[0]);
      if (other_line.size() > 1) {
        other_end_point_.push_back(other_line[other_line.size() - 1]);
      }
    }

    // calculate the distance between lines
    int point_ids_first = -1;
    int point_ids_second = -1;
    float min_dist = 10000;
    for (int i = 0; i < current_end_point_.size(); i++) {
      for (int j = 0; j < other_end_point_.size(); j++) {
        float distance = (current_end_point_[i].pt().x() - other_end_point_[j].pt().x()) *
                             (current_end_point_[i].pt().x() - other_end_point_[j].pt().x()) +
                         (current_end_point_[i].pt().y() - other_end_point_[j].pt().y()) *
                             (current_end_point_[i].pt().y() - other_end_point_[j].pt().y());
        if (distance < min_dist) {
          point_ids_first = i;
          point_ids_second = j;
          min_dist = distance;
        }
      }
    }

    // add one line below threshold
    if (min_dist < kDistanceThreshold) {
      lld::PointList2f left_line, right_line;
      int pos = point_ids_second / 2;
      lld::PointList2f adjacent_line = other_lines_[pos];
      other_lines_.erase(other_lines_.begin() + pos, other_lines_.begin() + pos + 1);
      if (point_ids_first == 0 && point_ids_second % 2 == 0) {
        for (int i = adjacent_line.size() - 1; i >= 0; i--) {
          left_line.push_back(adjacent_line[i]);
        }
        right_line = current_line;
      } else if (point_ids_first == 0 && point_ids_second % 2 == 1) {
        left_line = adjacent_line;
        right_line = current_line;
      } else if (point_ids_first == 1 && point_ids_second % 2 == 0) {
        left_line = current_line;
        right_line = adjacent_line;
      } else if (point_ids_first == 1 && point_ids_second % 2 == 1) {
        left_line = current_line;
        for (int i = adjacent_line.size() - 1; i >= 0; i--) {
          right_line.push_back(adjacent_line[i]);
        }
      }
      long_lines_.clear();
      lld::PointList2f tmp_lines;
      for (auto &other_line : other_lines_) {
        long_lines_.push_back(other_line);
      }

      for (auto &left_point : left_line) {
        if (tmp_lines.size() >= kMaxPointListLen) {
          break;
        }
        tmp_lines.push_back(left_point);
      }
      for (auto &right_point : right_line) {
        if (tmp_lines.size() >= kMaxPointListLen) {
          break;
        }
        tmp_lines.push_back(right_point);
      }
      long_lines_.push_back(tmp_lines);
    } else {
      final_lines_.push_back(current_line);
      long_lines_ = other_lines_;
    }
  }

  final_lines_.push_back(long_lines_[0]);
  res_lanes.resize(0);
  for (auto &line : final_lines_) {
    if (line.size() > res_lanes.size()) {
      res_lanes = line;
    }
  }
}

void TaskEdgeLld::SerializeSingleLine(lld::PointList2f &pred_lanes, lld::PointList2f &res_lanes) {
  // predict piecewise lines for linking
  lld::PointList2f each_lanes = pred_lanes;
  piecewise_lines_.clear();
  res_lanes.reserve(kMaxPointListLen);

  while (pred_lanes.size() > 0) {
    res_lanes.clear();
    // remove isolated points
    int point_nums = pred_lanes.size();
    for (int i = 0; i < point_nums; i++) {
      for (int j = 0; j < point_nums; j++) {
        if (i == j) {
          continue;
        }

        float distance = std::max(std::abs(pred_lanes[i].pt().x() - pred_lanes[j].pt().x()),
                                  std::abs(pred_lanes[i].pt().y() - pred_lanes[j].pt().y()));
        if (distance == 1) {
          res_lanes.push_back(pred_lanes[i]);
          break;
        }
      }
    }

    pred_lanes.clear();
    pred_lanes = res_lanes;

    if (pred_lanes.size() == 0) {
      break;
    }

    // get the point list ymax in a line
    float y_max = 0;
    for (auto &point : pred_lanes) {
      y_max = point.pt().y() > y_max ? point.pt().y() : y_max;
    }

    // select the point == ymax in a line
    lld::PointList2f selected_points;
    lld::PointList2f out_selected_points;
    // select the point != ymax in a line
    lld::PointList2f alternative_points;

    for (auto &point : pred_lanes) {
      if (selected_points.size() >= kMaxPointListLen || alternative_points.size() >= kMaxPointListLen) {
        break;
      }
      if (point.pt().y() == y_max && selected_points.size() == 0) {
        selected_points.push_back(point);
      } else {
        alternative_points.push_back(point);
      }
    }

    y_max -= 1;

    while (alternative_points.size() > 0) {
      // select the point y >= ymax in a line
      lld::PointList2f near_points;
      // select the point y < ymax in a line
      lld::PointList2f far_points;
      for (auto &point : alternative_points) {
        if (near_points.size() >= kMaxPointListLen || far_points.size() >= kMaxPointListLen) {
          break;
        }
        if (point.pt().y() >= y_max) {
          near_points.push_back(point);
        } else {
          far_points.push_back(point);
        }
      }

      if (near_points.size() == 0) {
        break;
      }

      // compare near_points and selected_points distances
      lld::PointList2f tmp_points = near_points;
      while (tmp_points.size() > 0) {
        // satisfied distance points to be added
        lld::PointList2f added_points;
        for (auto &n_point : tmp_points) {
          for (auto &s_point : selected_points) {
            float distance =
                std::max(std::abs(n_point.pt().x() - s_point.pt().x()), std::abs(n_point.pt().y() - s_point.pt().y()));
            if (distance == 1) {
              // compute_vertical_distance
              lld::PointList2f vertical_points;
              float vertical_distance;
              for (auto &dist_point : selected_points) {
                if (vertical_points.size() >= kMaxPointListLen) {
                  break;
                }
                if (n_point.pt().x() == dist_point.pt().x()) {
                  vertical_points.push_back(dist_point);
                }
              }

              if (vertical_points.size() == 0) {
                vertical_distance = 0;
              } else {
                vertical_distance = 10000;
                for (auto &v_pnt : vertical_points) {
                  float min_distance = std::abs(v_pnt.pt().y() - n_point.pt().y());
                  vertical_distance = min_distance < vertical_distance ? min_distance : vertical_distance;
                }
              }

              if (vertical_distance <= 1) {
                if (selected_points.size() >= kMaxPointListLen || added_points.size() >= kMaxPointListLen) {
                  break;
                }
                selected_points.push_back(n_point);
                added_points.push_back(n_point);
                break;
              }
            }
          }
        }

        if (added_points.size() == 0) {
          break;
        } else {
          if (added_points.size() < tmp_points.size()) {
            lld::PointList2f new_points;
            for (auto &n_point : tmp_points) {
              bool isInlist = false;
              for (auto &a_point : added_points) {
                if (a_point.pt().x() == n_point.pt().x() && a_point.pt().y() == n_point.pt().y()) {
                  isInlist = true;
                }
              }

              if (!isInlist) {
                if (new_points.size() >= kMaxPointListLen) {
                  break;
                }
                new_points.push_back(n_point);
              }
            }
            tmp_points = new_points;
          } else {
            tmp_points.clear();
          }
        }
      }

      if (near_points.size() == tmp_points.size()) {
        break;
      } else {
        alternative_points.clear();
        if (alternative_points.size() >= kMaxPointListLen) {
          break;
        }
        for (auto &t_points : tmp_points) {
          alternative_points.push_back(t_points);
        }
        for (auto &f_points : far_points) {
          alternative_points.push_back(f_points);
        }
        y_max -= 1;
      }
    }

    ExtendEndPoints(selected_points, each_lanes);
    // PickHighConfPoints(selected_points, out_selected_points);
    lld::PointList2f piecewise_line;
    // ArrangePoints2Line(out_selected_points, piecewise_line);
    ArrangePoints2Line(selected_points, piecewise_line);
    piecewise_lines_.push_back(piecewise_line);
    pred_lanes = alternative_points;
  }

  cv::Mat show = cv::Mat::ones(152 * 4, 240 * 4, CV_8UC3) * 255;
  for (int l = 0; l < piecewise_lines_.size(); l++) {
    for (int p = 0; p < piecewise_lines_[l].size(); p++) {
      show.at<cv::Vec3b>(piecewise_lines_[l][p].pt().y(), piecewise_lines_[l][p].pt().x()) = cv::Vec3b(l * 10, l * 10, l * 10);
    }
    cv::namedWindow("piece" + std::to_string(l), 0);
    cv::imshow("piece" + std::to_string(l), show);
  }
  cv::waitKey(0);

  if (piecewise_lines_.size() >= 1) {
    ConnectPiecewiseLines(piecewise_lines_, res_lanes);
  } else {
    res_lanes.clear();
  }
}

void TaskEdgeLld::ReverseLines(std::vector<lld::PointList2f> &lanes) {
  std::vector<lld::PointList2f> new_lanes;
  for (int l = 0; l < lanes.size(); l++) {
    auto res_lanes = lanes[l];
    if (res_lanes.size() > 0) {
      /*lld::PointList2f reverse_lanes;
      if (res_lanes[0].pt().y() < res_lanes[res_lanes.size() - 1].pt().y()) {
        for (int i = res_lanes.size() - 1; i >= 0; i--) {
          reverse_lanes.push_back(res_lanes[i]);
        }
        res_lanes = reverse_lanes;
      }*/

      // reverse line if points order opposites with points orientation
      lld::Point2fWithVariance pre_point = res_lanes[0];
      int right_order_num = 0;
      int wrong_order_num = 0;
      for (int i = 1; i < res_lanes.size(); i++) {
        lld::Point2fWithVariance curr_point = res_lanes[i];
        float line_orient = CalculatePointsOrientation(pre_point, curr_point);
        float orient = pre_point.pt().orientation();
        bool reverse = false;  // 代表反向是否反了
        if (orient != -1) {
          float orient_diff = fabs(line_orient - orient);
          if (orient_diff > 180) {
            orient_diff = 360 - orient_diff;
          }

          if (orient_diff > 90) {
            reverse = true;
          }
        }

        pre_point = curr_point;

        if (reverse) {
          wrong_order_num++;
        } else {
          right_order_num++;
        }
      }

      if (wrong_order_num > right_order_num) {
        lld::PointList2f reverse_lanes;
        for (int i = res_lanes.size() - 1; i >= 0; i--) {
          reverse_lanes.push_back(res_lanes[i]);
        }
        res_lanes = reverse_lanes;
      }

      new_lanes.push_back(res_lanes);
    }
  }

  lanes = new_lanes;
}

void TaskEdgeLld::SortIndices(const float *confs, const int num_boxes, std::vector<int> &idx) {
  int boxnum = num_boxes;
  idx.resize(boxnum);
  for (size_t i = 0; i != idx.size(); ++i) {
    idx[i] = i;
  }
  std::stable_sort(idx.begin(), idx.end(), [&confs](int i1, int i2) { return confs[i1] > confs[i2]; });
}

void TaskEdgeLld::ArrangePoints2Line(lld::PointList2f &selected_points, lld::PointList2f &res_points) {
  x_value_.resize(selected_points.size());
  y_value_.resize(selected_points.size());
  confidence_value_.resize(selected_points.size());
  line_embedding_value_.resize(selected_points.size());
  category_value_.resize(selected_points.size());
  orientation_value_.resize(selected_points.size());
  attr_value_.resize(selected_points.size());

  for (int i = 0; i < selected_points.size(); i++) {
    int index = int(selected_points[i].pt().y() + 0.5) * featuremap_width_ + int(selected_points[i].pt().x() + 0.5);
    int col = int(selected_points[i].pt().x() + 0.5);
    int row = int(selected_points[i].pt().y() + 0.5);
    int j = 0;
    for (j = 0; j < idx_list_.size(); j++) {
      if (idx_list_[j].x_ == col && idx_list_[j].y_ == row) {
        break;
      }
    }
    if (j < idx_list_.size()) {
      x_value_[i] = idx_list_[j].px_;
      y_value_[i] = idx_list_[j].py_;
      confidence_value_[i] = idx_list_[j].heatmap_value_;
      line_embedding_value_[i] = idx_list_[j].line_embedding_;
      category_value_[i] = idx_list_[j].category_;
      orientation_value_[i] = idx_list_[j].orientation_;
      attr_value_[i] = idx_list_[j].attr_;
    } else {
      LOG(INFO) << "LLD unfound AdHeatMapPoints2D";
    }
  }

  SortIndices(x_value_.data(), selected_points.size(), idx_);
  if (selected_points.size() >= kMaxPointListLen) {
    res_points.resize(kMaxPointListLen);
  } else {
    res_points.resize(selected_points.size());
  }

  for (int i = 0; i < res_points.size(); i++) {
    res_points[i] = lld::Point2fWithVariance({{std::min(x_value_[idx_[i]] * kAPLLDPointScale, image_width_ - 1),
                                               std::min(y_value_[idx_[i]] * kAPLLDPointScale, image_height_ - 1),
                                               confidence_value_[idx_[i]], line_embedding_value_[idx_[i]],
                                               category_value_[idx_[i]], orientation_value_[idx_[i]],
                                               attr_value_[idx_[i]]}, confidence_value_[idx_[i]]});
  }
}

bool sort_point_x_dim(lld::Point2fWithVariance point_a,  lld::Point2fWithVariance point_b) {
    return (point_a.pt().x() < point_b.pt().x() );
}

bool sort_point_y_dim(lld::Point2fWithVariance point_a,  lld::Point2fWithVariance point_b) {
   return (point_a.pt().y() < point_b.pt().y() );
}

void TaskEdgeLld::PickHighConfPoints(lld::PointList2f &selected_points, 
                                                                                lld::PointList2f & selected_points_out_){
  lld::PointList2f selected_points_sort_x_ (selected_points);
  lld::PointList2f selected_points_sort_y_ (selected_points);
  std::sort(selected_points_sort_x_.begin(), selected_points_sort_x_.end(),  sort_point_x_dim );
  std::sort(selected_points_sort_y_.begin(), selected_points_sort_y_.end(),  sort_point_y_dim );
  
  int x_min{ (int)selected_points_sort_x_[0].pt().x() };
  int y_min{ (int)selected_points_sort_y_[0].pt().y() };
  int x_max{ (int)selected_points_sort_x_[selected_points_sort_x_.size()-1].pt().x() };
  int y_max { (int)selected_points_sort_y_[selected_points_sort_y_.size()-1].pt().y() };
  // printf("x_min:%d, x_max:%d, y_min:%d, y_max:%d \n", x_min, x_max, y_min, y_max);
  int x_range{x_max-x_min};
  int y_range{y_max-y_min};
  int32_t size = selected_points.size();
  if(x_range > y_range){
    int32_t save_x_value {0}; //record max confidence point
    for(int32_t i =0; i < size; i ++){
      if( selected_points_sort_x_[i].pt().x() == selected_points_sort_x_[save_x_value].pt().x() ){
        if( selected_points_sort_x_[i].pt().conf() >  selected_points_sort_x_[save_x_value].pt().conf() ){
          save_x_value = i;
        }
      }
      else{
       selected_points_out_.push_back(selected_points_sort_x_[save_x_value]);
        save_x_value = i; 
      }

      if(i == size-1){
         selected_points_out_.push_back(selected_points_sort_x_[save_x_value]);
      }
    }
  }
  else{ // x_range < y_range
   int32_t save_y_value {0}; //record max confidence point
    for(int32_t i =0; i < size; i ++){ 
      if( selected_points_sort_y_[i].pt().y() == selected_points_sort_y_[save_y_value].pt().y() ){
        if( selected_points_sort_y_[i].pt().conf() >  selected_points_sort_y_[save_y_value].pt().conf() ){
          save_y_value = i;
        }
      }
      else{
       selected_points_out_.push_back(selected_points_sort_y_[save_y_value]);
        save_y_value = i;
      }

      if(i == size-1){
         selected_points_out_.push_back(selected_points_sort_y_[save_y_value]);
      }
    }
  }
}

void TaskEdgeLld::ExtendEndPoints(lld::PointList2f &selected_points, lld::PointList2f &pred_lanes) {
  int min_x = 10000;
  int max_x = 0;
  lld::PointList2f left_endpoints, right_endpoints;

  for (auto &s_point : selected_points) {
    if (left_endpoints.size() >= kMaxPointListLen || right_endpoints.size() >= kMaxPointListLen) {
      break;
    }
    if (s_point.pt().x() == min_x) {
      left_endpoints.push_back(s_point);
    } else if (s_point.pt().x() <= min_x) {
      left_endpoints.clear();
      left_endpoints.push_back(s_point);
      min_x = s_point.pt().x();
    }
    if (s_point.pt().x() == max_x) {
      right_endpoints.push_back(s_point);
    } else if (s_point.pt().x() > max_x) {
      right_endpoints.clear();
      right_endpoints.push_back(s_point);
      max_x = s_point.pt().x();
    }
  }

  // add points around left end point
  for (auto &l_point : left_endpoints) {
    int l_point_x = l_point.pt().x();
    int l_point_y = l_point.pt().y();
    for (int i = 0; i < pred_lanes.size(); i++) {
      if (l_point_x - 1 == pred_lanes[i].pt().x() && l_point_y == pred_lanes[i].pt().y()) {
        if (selected_points.size() >= kMaxPointListLen) {
          break;
        }
        selected_points.push_back(lld::Point2fWithVariance({{l_point_x - 1, l_point_y}, 0}));
        l_point_x -= 1;
        i = -1;
      }
    }
  }

  // add points around right end point
  for (auto &r_point : right_endpoints) {
    int r_point_x = r_point.pt().x();
    int r_point_y = r_point.pt().y();
    for (int i = 0; i < pred_lanes.size(); i++) {
      if (r_point_x + 1 == pred_lanes[i].pt().x() && r_point_y == pred_lanes[i].pt().y()) {
        if (selected_points.size() >= kMaxPointListLen) {
          break;
        }
        selected_points.push_back(lld::Point2fWithVariance({{r_point_x + 1, r_point_y}, 0}));
        r_point_x += 1;
        i = -1;
      }
    }
  }
}
