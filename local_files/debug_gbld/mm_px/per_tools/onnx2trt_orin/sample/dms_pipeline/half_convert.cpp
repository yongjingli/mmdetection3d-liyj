
#include "half_convert.h"
#include <iostream>
void half2float::CreatH2FTable(float *bigtable)  // for half to float
{
  for (unsigned int i = 0; i < 65536; i++) {
    bigtable[i] = h2f_internal(i);
  }
}
bool half2float::initialized_ = false;  //
float *half2float::bigtable_ = nullptr;
int half2float::user_count_ = 0;
half2float::half2float()  // init the tables
{
  if (!initialized_) {
    bigtable_ = new float[65536];
    CreatH2FTable(bigtable_);
    initialized_ = true;
  }
  ++user_count_;
}

half2float::~half2float() {
  --user_count_;
  if (user_count_ == 0) {
    // std::cout << "delete bigtable_" << std::endl;
    delete[] bigtable_;
  }
}
float half2float::convert(const unsigned short &h) { return bigtable_[h]; }
// for float to half
void float2half::CreatF2HTable(unsigned int *basetable, unsigned int *shifttable) {
  unsigned int i;
  int e;
  for (i = 0; i < 256; ++i) {
    e = i - 127;
    if (e < -24) {  // Very small numbers map to zero
      basetable[i | 0x000] = 0x0000;
      basetable[i | 0x100] = 0x8000;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    } else if (e < -14) {  // Small numbers map to denorms
      basetable[i | 0x000] = (0x0400 >> (-e - 14));
      basetable[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
      shifttable[i | 0x000] = -e - 1;
      shifttable[i | 0x100] = -e - 1;
    } else if (e <= 15) {  // Normal numbers just lose precision
      basetable[i | 0x000] = ((e + 15) << 10);
      basetable[i | 0x100] = ((e + 15) << 10) | 0x8000;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    } else if (e < 128) {  // Large numbers map to Infinity
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 24;
      shifttable[i | 0x100] = 24;
    } else {  // Infinity and NaN's stay Infinity and NaN's
      basetable[i | 0x000] = 0x7C00;
      basetable[i | 0x100] = 0xFC00;
      shifttable[i | 0x000] = 13;
      shifttable[i | 0x100] = 13;
    }
  }
}
float2half::float2half() {
  basetable_ = new unsigned int[1024];
  shifttable_ = new unsigned int[1024];
  CreatF2HTable(basetable_, shifttable_);
}
float2half::~float2half() {
  delete[] shifttable_;
  delete[] basetable_;
}
unsigned short float2half::convert(const float &data) {
  FP32 fp32;
  fp32.v_f = data;
  unsigned short h =
      basetable_[(fp32.v_i >> 23) & 0x1ff] + ((fp32.v_i & 0x007fffff) >> shifttable_[(fp32.v_i >> 23) & 0x1ff]);
  return h;
}
int2half::int2half()  // creat table
{
  datatable_ = new unsigned short[256];
  float2half *float2half_tool = new float2half;
  for (int i = 0; i < 255; i++) {
    float data = i - 128.0;
    datatable_[i] = float2half_tool->convert(data);
  }
  delete float2half_tool;
}
int2half::~int2half() { delete[] datatable_; }
unsigned short int2half::convert(const int8_t &data) {
  int index = data + 128;
  return (datatable_[index]);
}

uint2half::uint2half()  // creat table
{
  datatable_ = new unsigned short[256];
  float2half *float2half_tool = new float2half;
  for (int i = 0; i < 256; i++) {
    datatable_[i] = float2half_tool->convert(static_cast<float>(i));
  }
  delete float2half_tool;
}
uint2half::~uint2half() { delete[] datatable_; }
unsigned short uint2half::convert(const uint8_t &data) {
  // int index = static_cast<int>(data);
  return (datatable_[data]);
}