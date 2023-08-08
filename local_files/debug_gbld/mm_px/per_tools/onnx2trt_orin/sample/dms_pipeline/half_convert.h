#ifndef HALF_CONVERT_H_
#define HALF_CONVERT_H_
#include <stdint.h>

typedef union {
  float v_f;
  unsigned int v_i;
} FP32;

inline float h2f_internal(unsigned short a) {
  unsigned int sign = (unsigned int)(a & 0x8000) << 16;
  int aexp = (a >> 10) & 0x1f;
  unsigned int mantissa = a & 0x3ff;
  if (aexp == 0x1f)  // && ieee)
    return sign | 0x7f800000 | (mantissa << 13);
  if (aexp == 0) {
    int shift;
    if (mantissa == 0) return sign;
    shift = __builtin_clz(mantissa) - 21;
    mantissa <<= shift;
    aexp = -shift;
  }
  FP32 outfp;
  outfp.v_i = sign | (((aexp + 0x70) << 23) + (mantissa << 13));
  return outfp.v_f;
}

// h2f_tool(use one tables) cost 10 ms every 2000000*5 times
class half2float {
 private:
  static bool initialized_;
  static float *bigtable_;
  static int user_count_;
  void CreatH2FTable(float *bigtable);

 public:
  half2float();

  float convert(const unsigned short &a);
  ~half2float();
};

class float2half {
 private:
  unsigned int *shifttable_;
  unsigned int *basetable_;
  void CreatF2HTable(unsigned int *basetable, unsigned int *shifttable);

 public:
  float2half();
  unsigned short convert(const float &data);
  ~float2half();
};

class int2half  // int8 to float
{
 private:
  unsigned short *datatable_;

 public:
  int2half();
  unsigned short convert(const int8_t &data);
  ~int2half();
};
class uint2half  // uint8 to float
{
 private:
  unsigned short *datatable_;

 public:
  uint2half();
  unsigned short convert(const uint8_t &data);
  ~uint2half();
};
#endif  // HALF_CONVERT_H_