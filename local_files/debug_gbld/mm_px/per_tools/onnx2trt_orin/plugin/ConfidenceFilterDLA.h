/*
 * Copyright (c) 2019, Xiaopeng. All rights reserved.
 * Create by caizw @ 2019.9.12
 * Use the onnx operator "DecodeNMS" for operator decode & nms

 */

#ifndef __CONFIDENCE_FILTER_DLA_HPP__
#define __CONFIDENCE_FILTER_DLA_HPP__

#include "common.h"

// For DLA fp16-chw16 or int8-chw32, similar to hwc if c<=16
#define MAXINPUTNUM 16
typedef struct stFilterParam {
  const void *_d_input_bufs[MAXINPUTNUM];
  void *outputs[MAXINPUTNUM];
  int4 _input_dims[MAXINPUTNUM];
  int _input_size[MAXINPUTNUM];
  int _count[MAXINPUTNUM];
  int _conf_offset[MAXINPUTNUM];
  float thresholds[MAXINPUTNUM];
  int fpn_shape[MAXINPUTNUM];
  float out_scale[MAXINPUTNUM];  // Output scale for int8, same as those in CalibrationTable
  int mode;
  int max_num;
  int input_num;
  int data_type;  // 0: fp32, 1: fp16, 2: int8
} FilterParam;

typedef struct ldFilterParam {
  const void *_d_input_bufs[MAXINPUTNUM];
  void *outputs[MAXINPUTNUM];
  int  *_count[MAXINPUTNUM];

  int4 _input_dims[MAXINPUTNUM];
  int _input_size[MAXINPUTNUM];
  int _conf_offset[MAXINPUTNUM];
  float thresholds[MAXINPUTNUM];
  int fpn_shape[MAXINPUTNUM];
  float out_scale[MAXINPUTNUM];  // Output scale for int8, same as those in CalibrationTable
  int mode;
  int max_num;
  int input_num;
  int data_type;  // 0: fp32, 1: fp16, 2: int8
 
}LidarFilterParam;

extern "C" int ConfidenceFilterForDLA(int batchSize, FilterParam *param, cudaStream_t stream);

extern "C" int ConfidenceFilterForLidarDLA(LidarFilterParam *param, cudaStream_t stream);

#endif //__CONFIDENCE_FILTER_DLA_HPP__
