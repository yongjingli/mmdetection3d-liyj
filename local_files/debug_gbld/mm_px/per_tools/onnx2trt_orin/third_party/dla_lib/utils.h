/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstdint>
#include <string>
#include "cLogger.h"
#include "NvInfer.h"

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#define CHECK_FAIL(condition, message, ...)               \
  do {                                                    \
    if (!(condition)) {                                   \
      LOG_ERR("%s Fail\n", message, ##__VA_ARGS__);       \
      goto fail;                                          \
    } else {                                              \
      LOG_DBG("%s Successful\n", message, ##__VA_ARGS__); \
    }                                                     \
  } while (0)

#define PROPAGATE_ERROR_FAIL(condition, message, ...)     \
  do {                                                    \
    if (!(condition)) {                                   \
      LOG_ERR("%s Fail\n", message, ##__VA_ARGS__);       \
      status = NVMEDIA_STATUS_ERROR;                      \
      goto fail;                                          \
    } else {                                              \
      LOG_DBG("%s Successful\n", message, ##__VA_ARGS__); \
    }                                                     \
  } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            std::cout << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            fflush(stdout);                                                                                            \
        }                                                                                                              \
    } while (0)


#define CHECK_NVMEDIA_STATUS(status, val) CHECK_RETURN_W_MSG(status, val, "")
using namespace nvinfer1;

class HostMemory
{
public:
    HostMemory() = delete;
    virtual void* data() const noexcept
    {
        return mData;
    }
    virtual std::size_t size() const noexcept
    {
        return mSize;
    }
    virtual DataType type() const noexcept
    {
        return mType;
    }
    virtual ~HostMemory() {}

protected:
    HostMemory(std::size_t size, DataType type)
        : mSize(size)
        , mType(type)
    {
    }
    void* mData;
    std::size_t mSize;
    DataType mType;
};

template <typename ElemType, DataType dataType>
class TypedHostMemory : public HostMemory
{
public:
    TypedHostMemory(std::size_t size)
        : HostMemory(size, dataType)
    {
        mData = new ElemType[size];
    };
    ~TypedHostMemory() noexcept
    {
        delete[](ElemType*) mData;
    }
    ElemType* raw() noexcept
    {
        return static_cast<ElemType*>(data());
    }
};

using ByteMemory = TypedHostMemory<uint8_t, DataType::kINT8>;

uint8_t *readFileToMemory(std::string fileName, uint64_t &fileSize);
char *readBinFile(std::string fileName, uint32_t &fileSize);
void PrintLocalTime();  // add by chenlm
void writeToFile(const char *input_buffer, uint32_t size, std::string filename);
bool readBinToBuffer(std::string fileName, char *pBuffer, uint32_t size);

#endif  // end of _UTILS_H_
