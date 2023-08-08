/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>

#include "tensor.h"
#include "utils.h"

#define MODULE "DlaTensor"

Tensor::Tensor(NvMediaDevice *device) : m_pDevice(device), m_pTensor(nullptr) {}

Tensor::~Tensor() {
  m_pDevice = nullptr;

  if (m_pTensor) {
    NvMediaTensorDestroy(m_pTensor);
  }
}

NvMediaStatus Tensor::FillNvSciBufTensorAttrs(NvMediaDevice *device, NvMediaTensorAttr tensorAttrs[], uint32_t numAttrs,
                                              NvSciBufAttrList attr_h) {
  return NvMediaTensorFillNvSciBufAttrs(device, tensorAttrs, numAttrs, 0, attr_h);
}

NvMediaStatus Tensor::Create(NvSciBufObj bufObj) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  status = NvMediaTensorCreateFromNvSciBuf(m_pDevice, bufObj, &m_pTensor);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorCreateFromNvSciBuf");

fail:
  return status;
}

NvMediaStatus Tensor::SetData(uint8_t value) {
  m_status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &m_TensorMap);
  CHECK_FAIL(m_status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

  memset(m_TensorMap.mapping, value, m_TensorMap.size);
  NvMediaTensorUnlock(m_pTensor);

fail:
  return m_status;
}

// Fill tensor with data from buffer
NvMediaStatus Tensor::FillDataIntoTensor(uint32_t size, void *p) {
  m_status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &m_TensorMap);
  CHECK_FAIL(m_status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");
  // CHECK_FAIL(m_InputTensorMap.size >= size, "Tensor size check");
  if (NULL != p && p != m_TensorMap.mapping) {
    memcpy(m_TensorMap.mapping, p, size);
  }
  NvMediaTensorUnlock(m_pTensor);

fail:
  return m_status;
}

NvMediaStatus Tensor::CopyDataFromTensor(uint32_t size, void *p) {
  m_status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_READ, &m_TensorMap);
  CHECK_FAIL(m_status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

  if (NULL != p && p != m_TensorMap.mapping) {
    memcpy(p, m_TensorMap.mapping, size);
  }

  NvMediaTensorUnlock(m_pTensor);
fail:
  return m_status;
}

NvMediaTensor *Tensor::GetTensorPtr() const { return m_pTensor; }

void *Tensor::GetTensorMapPtr() { return m_TensorMap.mapping; }

uint32_t Tensor::GetTensorMapSize() { return m_TensorMap.size; }

NvMediaStatus Tensor::GetStatus() {
  m_status = NvMediaTensorGetStatus(m_pTensor, NVMEDIA_TENSOR_TIMEOUT_INFINITE, &m_taskStatus);
  CHECK_FAIL(m_status == NVMEDIA_STATUS_OK, "NvMediaTensorGetStatus");
  if (m_taskStatus.status != NVMEDIA_STATUS_OK) {
    m_status = m_taskStatus.status;
    LOG_ERR("Engine returned error.\n");
    goto fail;
  }

fail:
  return m_status;
}

NvMediaStatus Tensor::writeToFile(std::string filename) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvMediaTensorSurfaceMap tensorMap{};
  std::ofstream file(filename, std::ios::binary);
  char *pBuffer;

  // Read output
  status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_READ, &tensorMap);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");
  pBuffer = new char[tensorMap.size];
  memcpy(pBuffer, tensorMap.mapping, tensorMap.size);

  file.write(pBuffer, tensorMap.size);
  file.close();
  delete[] pBuffer;
  NvMediaTensorUnlock(m_pTensor);
fail:
  return status;
}

NvMediaStatus Tensor::CompareWithRef(uint32_t size, void *p) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvMediaTensorSurfaceMap tensorMap{};

  // Read output and check results.
  status = NvMediaTensorLock(m_pTensor, NVMEDIA_TENSOR_ACCESS_READ, &tensorMap);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorLock");

  if (0 != memcmp(tensorMap.mapping, p, size)) {
    LOG_ERR("Output does not match expected\n");
    status = NVMEDIA_STATUS_ERROR;
    NvMediaTensorUnlock(m_pTensor);
    goto fail;
  } else {
    LOG_INFO("Compare with ref data: pass\n");
  }

  NvMediaTensorUnlock(m_pTensor);

fail:
  return status;
}
