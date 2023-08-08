/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <string>

#include "nvmedia_core.h"
#include "nvmedia_tensor.h"
#include "nvmedia_tensor_nvscibuf.h"

//! Class for create and destroy NvMedia Tensor from NvSciBuf
class Tensor {
 public:
  static NvMediaStatus FillNvSciBufTensorAttrs(NvMediaDevice *device, NvMediaTensorAttr tensorAttrs[],
                                               uint32_t numAttrs, NvSciBufAttrList attr_h);

  Tensor(NvMediaDevice *device);

  NvMediaStatus Create(NvSciBufObj bufObj);

  // Fill tensor with single value
  NvMediaStatus SetData(uint8_t value);

  NvMediaStatus FillDataIntoTensor(uint32_t size, void *p);
  NvMediaStatus CopyDataFromTensor(uint32_t size, void *p);

  // Fill tensor with data from pgm image file
  virtual NvMediaStatus FillDataIntoTensor(std::string pgmImageFileName) { return NVMEDIA_STATUS_NOT_SUPPORTED; }

  NvMediaStatus m_status = NVMEDIA_STATUS_OK;

  NvMediaTensor *GetTensorPtr() const;

  NvMediaStatus GetStatus();

  NvMediaStatus CompareWithRef(uint32_t size, void *p);
  NvMediaStatus writeToFile(std::string filename);

  void *GetTensorMapPtr();
  uint32_t GetTensorMapSize();

  virtual ~Tensor();

 protected:
  NvMediaDevice *m_pDevice;

  NvMediaTensor *m_pTensor;

 private:
  NvMediaTensorSurfaceMap m_TensorMap{};
  NvMediaTensorTaskStatus m_taskStatus{};
};

#endif  // end of _TENSOR_H_
