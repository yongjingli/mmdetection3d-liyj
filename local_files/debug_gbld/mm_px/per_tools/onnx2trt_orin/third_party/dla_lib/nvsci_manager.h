/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#pragma once

#ifndef _NVSCI_MANAGER_
#define _NVSCI_MANAGER_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <memory>
#include "nvmedia_core.h"

#include "cudaNvSciSignal.h"
#include "utils.h"

class Dla;

class NvSciManager final {
 public:
  NvSciManager(NvSciBufModule& buffModule, Dla* m_upDla, int cudaDeviceId, cudaStream_t stream);
  ~NvSciManager() = default;

  CudaNvSciSignal* GetCudaNvSciSignaler() { return cudaNvSciSignaler_.get(); }

  NvSciSyncFence* GetCudaSignalerFence() { return m_fence; }

 private:
  void initNvSci();

  NvMediaStatus SetupNvSciSync(Dla* m_upDla, CudaNvSciSignal* cudaNvSciSignaler);

  NvMediaStatus CreateSyncObjFromAttrList(NvSciSyncAttrList list1, NvSciSyncAttrList list2, NvSciSyncObj* syncObj);

  inline void FreeSyncAttrList(NvSciSyncAttrList list) {
    if (list != nullptr) {
      NvSciSyncAttrListFree(list);
      list = nullptr;
    }
  }

  NvSciSyncModule syncModule_;
  // NvSciBufModule buffModule;
  NvSciSyncFence* m_fence;

  NvSciSyncObj cudaSignalerSyncObj_;

  std::unique_ptr<CudaNvSciSignal> cudaNvSciSignaler_;
};

#endif