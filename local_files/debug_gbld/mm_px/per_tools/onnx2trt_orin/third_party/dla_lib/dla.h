/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _DLA_H_
#define _DLA_H_

#include <array>
#include <memory>
#include <vector>

#include "nvmedia_core.h"
#include "nvmedia_dla.h"
#include "nvmedia_dla_nvscisync.h"
#include "nvscisync.h"

#include "tensor.h"

//! Dla class
//! Dla class abstract NvMediaDla APIs and provide functions to load loadable and
//! execute loadable with provided input data.

class Dla final {
 public:
  NvMediaStatus m_status = NVMEDIA_STATUS_OK;
  static NvMediaStatus GetDlaVersion(NvMediaVersion *version);

  static NvMediaStatus PingById(const uint32_t dlaId);

  static std::unique_ptr<Dla> Create();

  ~Dla();

  NvMediaStatus Init(uint32_t dlaId, uint32_t numTasks);

  //! One Dla class can hold only one loadable.
  NvMediaStatus AddLoadable(std::string profileName, uint32_t &loadableIndex);

  NvMediaStatus GetDesc(uint32_t loadableIndex, std::vector<NvMediaDlaTensorDescriptor> &vInputTensorDesc,
                        std::vector<NvMediaDlaTensorDescriptor> &vOutputTensorDesc);

  NvMediaStatus DataRegister(uint32_t loadableIndex, Tensor *tensor);

  NvMediaStatus DataUnregister(uint32_t loadableIndex, Tensor *tensor);

  NvMediaStatus RemoveLoadable(uint32_t loadableIndex);

  NvMediaStatus Submit(uint32_t loadableIndex, std::vector<Tensor *> &vpInputTensor,
                       std::vector<Tensor *> &vpOutputTensor);

  // SciSync related api
  NvMediaStatus GetAttrList(NvSciSyncModule module, NvSciSyncAttrList &attrList, NvMediaNvSciSyncClientType syncType);

  NvMediaStatus RegisterSyncObj(NvMediaNvSciSyncObjType syncObjType, NvSciSyncObj syncObj);

  NvMediaStatus UnRegisterSyncObj(NvSciSyncObj syncObj);

  NvMediaStatus SetEOFSyncObj(NvSciSyncObj syncObj);

  NvMediaStatus InsertPreSciFences(NvSciSyncFence *EOFfence);

  NvMediaStatus GetEOFSciFences(NvSciSyncObj eofSyncObj, NvSciSyncFence *EOFfence);

 protected:
  NvMediaStatus PrintTensorDesc(NvMediaDlaTensorDescriptor *tensorDesc);

 private:
  Dla(NvMediaDla *m_pDla);

  NvMediaDla *m_pDla;

  std::vector<NvMediaDlaLoadable *> m_vLoadables;

  static const std::size_t MAX_NUM_OF_DLA_DATA = 10;

  std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aInputDlaData;

  std::array<NvMediaDlaData, MAX_NUM_OF_DLA_DATA> m_aOutputDlaData;
  NvMediaDlaArgs m_inputArgs{};
  NvMediaDlaArgs m_outputArgs{};
};

#endif  // END OF _DLA_H_
