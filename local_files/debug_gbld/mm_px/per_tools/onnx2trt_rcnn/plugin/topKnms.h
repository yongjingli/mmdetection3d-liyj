#ifndef CAFFE2_OPERATORS_TOP_K_H_
#define CAFFE2_OPERATORS_TOP_K_H_

int sortByKey(void *d_temp_storage, size_t &temp_storage_bytes,
              const float *d_keys_in, float *d_keys_out, const int *d_values_in,
              int *d_values_out, int num_items, cudaStream_t stream);

void nms(int *keep_out, int *num_out, const float *boxes_host, int boxes_num,
         int boxes_dim, float nms_overlap_thresh, int *order,
         cudaStream_t stream);

int GetShiftedAnchors(int threadn, int im_height, int im_width, int sc_height,
                      int sc_width, int PM, int feat_stride, int *order,
                      float *bbox_deltas_f, float *im_i_boxes_f,
                      cudaStream_t stream, bool clip = true);

void SetCudaSymbol(int acNum, float *anchorBuf, int anNum);

#endif  // CAFFE2_OPERATORS_TOP_K_H_
