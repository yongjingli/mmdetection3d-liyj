
#include "landmark.h"
Landmark::Landmark() { h2f_tool = std::make_unique<half2float>(); }
void Landmark::ExtractLandmark(uint16_t *data[], const int out_idx, float *output) {
  for (int i = 0; i < g_lm_number * 2; i++) {
    output[i] = h2f_tool->convert(data[out_idx][i]);
  }
}
