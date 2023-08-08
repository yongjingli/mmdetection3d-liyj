#ifndef LANDMARK_H_
#define LANDMARK_H_
#include <memory>
#include "config.h"
#include "half_convert.h"
class Landmark {
 private:
  std::unique_ptr<half2float> h2f_tool;

 public:
  Landmark();
  ~Landmark(){};
  void ExtractLandmark(uint16_t *data[], const int out_idx, float *output);
};
#endif  // LANDMARK_H_