/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _DLATOOL_H_
#define _DLATOOL_H_

#include <memory>
#include <string>

#include "dla.h"
#include "nvsci_manager.h"
#include "onnxtrt.h"
#include "tensor.h"

//! Class to test runtime mode

class DlaTool final {
  // friend class DlaWorker;

 public:
  DlaTool(uint32_t dlaId, uint32_t numTasks, std::string profileName, bool IsPingTest = false,
          cudaStream_t stream = nullptr);

  ~DlaTool();

  NvMediaStatus SetUp();

  NvMediaStatus Run(void *input_buffer, uint32_t input_size, int inputType, void *output_buffer, uint32_t output_size,
                    int outType);
  void *GetInputPointer(int index);
  void *GetOutPointer(int index);
  uint32_t GetOutSize(int index);
  uint32_t GetInputSize(int index);

  int GetBufferInfo(EngineBuffer bufferInfo[]);

  // int CudaSignal();

 protected:
  NvMediaStatus InitNvSciBuf(void);

  void DeinitNvSciBuf(void);

  NvMediaStatus ReconcileAndAllocSciBufObj(NvMediaTensorAttr tensorAttrs[], uint32_t numAttrs, NvSciBufObj *sciBuf);

  NvMediaStatus ReconcileAndAllocSciBufObj(NvMediaTensorAttr tensorAttrs[], uint32_t numAttrs,
                                           NvSciBufAttrList attrList, NvSciBufObj *sciBuf);

 private:
  uint32_t m_dlaId;

  uint32_t m_numTasks;

  std::string m_profileName;
  const bool CheckStatus{true};

  bool m_isPingTest;

  uint32_t m_loadableIndex;

  NvMediaDevice *m_device = nullptr;

  std::unique_ptr<Dla> m_upDla;

  std::vector<NvSciBufObj> m_pInputTensorScibuf;

  std::vector<Tensor *> m_vupInputTensor;

  std::vector<void *> m_vupInputBufferGPU;

  std::vector<NvSciBufObj> m_pOutputTensorScibuf;

  std::vector<Tensor *> m_vupOutputTensor;

  std::vector<void *> m_vupOutputBufferGPU;

  NvSciBufModule m_NvscibufModule = nullptr;

  std::vector<NvMediaDlaTensorDescriptor> m_InputTensorDesc;
  std::vector<NvMediaDlaTensorDescriptor> m_OutputTensorDesc;

  std::unique_ptr<NvSciManager> m_nvsci_manager;

  cudaStream_t cuda_stream_;
};

#endif  // end of _DLATOOL_H_