/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "utils.h"

void PrintLocalTime()  // add by chenlm
{
  // prepend timestamp
  std::time_t timestamp = std::time(nullptr);
  tm *tm_local = std::localtime(&timestamp);
  std::cout << "[";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mon << "/";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
  std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
  std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
}

uint8_t *readFileToMemory(std::string fileName, uint64_t &fileSize) {
  char *pBuffer{nullptr};

  std::ifstream is(fileName, std::ifstream::binary | std::ios::ate);
  CHECK_FAIL(is, "Open filename = %s", fileName.c_str());

  fileSize = is.tellg();

  pBuffer = new char[fileSize];
  CHECK_FAIL(pBuffer != nullptr, "alloc buffer");

  is.seekg(0, is.beg);

  is.read(pBuffer, fileSize);
  is.close();

fail:
  return reinterpret_cast<uint8_t *>(pBuffer);
}

char *readBinFile(std::string fileName, uint32_t &fileSize) {
  char *pBuffer{nullptr};

  std::ifstream is(fileName, std::ifstream::binary | std::ios::ate);
  CHECK_FAIL(is, "Open filename = %s", fileName.c_str());

  fileSize = is.tellg();

  pBuffer = new char[fileSize];
  CHECK_FAIL(pBuffer != nullptr, "alloc buffer");

  is.seekg(0, is.beg);

  is.read(pBuffer, fileSize);
  is.close();

fail:
  return pBuffer;
}

bool readBinToBuffer(std::string fileName, char *pBuffer, uint32_t size) {
  uint32_t fileSize;
  std::ifstream is(fileName, std::ifstream::binary | std::ios::ate);

  CHECK_FAIL(pBuffer != nullptr, "buffer is null");
  CHECK_FAIL(is, "Open filename = %s", fileName.c_str());

  fileSize = is.tellg();
  // LOG_INFO("fileSize :%d\n",fileSize);
  // LOG_INFO("size  :%d\n",size);
  CHECK_FAIL(fileSize <= size, "buffer size is too small");

  is.seekg(0, is.beg);

  is.read(pBuffer, fileSize);

  is.close();
  return true;
fail:
  return false;
}

void writeToFile(const char *input_buffer, uint32_t size, std::string filename) {
  std::ofstream file(filename, std::ios::binary);
  file.write(input_buffer, size);
  file.close();
}
