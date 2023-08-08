#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cassert>
#include <vector>
#include "NvInfer.h"
using namespace nvinfer1;

#ifndef CHECK_CUDA
#define CHECK_CUDA(status)                                     \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != cudaSuccess)                                \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)
#endif

extern "C" int TRT_DEBUGLEVEL; // 4:VERBOSE, 3:DEBUG, 2:INFO, 1:WARN, 0:ERROR
#ifndef DPRINTF
#define DPRINTF(level, x...)           \
    do                                 \
    {                                  \
        if ((level) <= TRT_DEBUGLEVEL) \
        {                              \
            printf(x);                 \
        }                              \
    } while (0)
#endif

#define cudaDebugError(LEVEL) __cudaDebugError(LEVEL, stream, __FILE__, __LINE__)

inline void __cudaDebugError(int level, cudaStream_t stream, const char* file, const int line) {
  if ((level) <= TRT_DEBUGLEVEL) {
    cudaStreamSynchronize(stream);
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "onnxtrt cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    }
  }
  return;
}

extern "C" int convertCPU_GPU(void *pCPU, void *pGPU, int size, int type, int direction, cudaStream_t stream,
                   void *pBuffer);
inline unsigned int elementSize(DataType t) {
  switch (t) {
    case DataType::kINT32:
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
  }
  return 0;
}
template <typename T>
void write(char *&buffer, const T &val) 
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void write(char *&buffer, T *data, int count) 
{
    memcpy(buffer, data, sizeof(T) * count);
    buffer += sizeof(T) * count;
}

template <typename T>
T read(const char *&buffer) 
{
    T val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
    return val;
}

template <typename T>
void read(const char *&buffer, T *data, int count) 
{
    memcpy(data, buffer, sizeof(T) * count);
    buffer += sizeof(T) * count;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define CUDA_2D_KERNEL_BLOCK_LOOP(i, n, j, m)          \
  for (size_t i = blockIdx.x; i < (n); i += gridDim.x) \
    for (size_t j = blockIdx.y; j < (m); j += gridDim.y)


inline size_t getAlignedSize(size_t origin_size, size_t aligned_number = 16) {
  return size_t((origin_size + aligned_number - 1) / aligned_number) *
         aligned_number;
}


template <typename T>
inline void serialize_value(void** buffer, T const& value);

template <typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size,
                              T* value);

namespace {

template <typename T, class Enable = void>
struct Serializer {};

template <typename T>
struct Serializer<T, typename std::enable_if<std::is_arithmetic<T>::value ||
                                             std::is_enum<T>::value ||
                                             std::is_pod<T>::value>::type> {
  static size_t serialized_size(T const& value) { return sizeof(T); }
  static void serialize(void** buffer, T const& value) {
    ::memcpy(*buffer, &value, sizeof(T));
    reinterpret_cast<char*&>(*buffer) += sizeof(T);
  }
  static void deserialize(void const** buffer, size_t* buffer_size, T* value) {
    assert(*buffer_size >= sizeof(T));
    ::memcpy(value, *buffer, sizeof(T));
    reinterpret_cast<char const*&>(*buffer) += sizeof(T);
    *buffer_size -= sizeof(T);
  }
};

template <>
struct Serializer<const char*> {
  static size_t serialized_size(const char* value) { return strlen(value) + 1; }
  static void serialize(void** buffer, const char* value) {
    ::strcpy(static_cast<char*>(*buffer), value);
    reinterpret_cast<char*&>(*buffer) += strlen(value) + 1;
  }
  static void deserialize(void const** buffer, size_t* buffer_size,
                          const char** value) {
    *value = static_cast<char const*>(*buffer);
    size_t data_size = strnlen(*value, *buffer_size) + 1;
    assert(*buffer_size >= data_size);
    reinterpret_cast<char const*&>(*buffer) += data_size;
    *buffer_size -= data_size;
  }
};

template <typename T>
struct Serializer<std::vector<T>,
                  typename std::enable_if<std::is_arithmetic<T>::value ||
                                          std::is_enum<T>::value ||
                                          std::is_pod<T>::value>::type> {
  static size_t serialized_size(std::vector<T> const& value) {
    return sizeof(value.size()) + value.size() * sizeof(T);
  }
  static void serialize(void** buffer, std::vector<T> const& value) {
    serialize_value(buffer, value.size());
    size_t nbyte = value.size() * sizeof(T);
    ::memcpy(*buffer, value.data(), nbyte);
    reinterpret_cast<char*&>(*buffer) += nbyte;
  }
  static void deserialize(void const** buffer, size_t* buffer_size,
                          std::vector<T>* value) {
    size_t size;
    deserialize_value(buffer, buffer_size, &size);
    value->resize(size);
    size_t nbyte = value->size() * sizeof(T);
    assert(*buffer_size >= nbyte);
    ::memcpy(value->data(), *buffer, nbyte);
    reinterpret_cast<char const*&>(*buffer) += nbyte;
    *buffer_size -= nbyte;
  }
};

}  // namespace

template <typename T>
inline size_t serialized_size(T const& value) {
  return Serializer<T>::serialized_size(value);
}

template <typename T>
inline void serialize_value(void** buffer, T const& value) {
  return Serializer<T>::serialize(buffer, value);
}

template <typename T>
inline void deserialize_value(void const** buffer, size_t* buffer_size,
                              T* value) {
  return Serializer<T>::deserialize(buffer, buffer_size, value);
}
#endif //__COMMON_HPP__
