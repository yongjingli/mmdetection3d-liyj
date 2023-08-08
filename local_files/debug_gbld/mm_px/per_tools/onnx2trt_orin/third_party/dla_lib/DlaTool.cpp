/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "DlaTool.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>

#include "utils.h"

DlaTool::DlaTool(uint32_t dlaId, uint32_t numTasks, std::string profileName, bool IsPingTest, cudaStream_t stream)
    : m_dlaId(dlaId),
      m_numTasks(numTasks),
      m_profileName(profileName),
      m_isPingTest(IsPingTest),
      m_loadableIndex(0),
      cuda_stream_(stream) {}

DlaTool::~DlaTool() {
  if (m_isPingTest) {
    return;
  }

  if (m_device) {
    NvMediaDeviceDestroy(m_device);
  }

  for (auto i = 0u; i < m_vupInputTensor.size(); i++) {
    m_upDla->DataUnregister(m_loadableIndex, m_vupInputTensor[i]);
  }

  for (auto i = 0u; i < m_vupOutputTensor.size(); i++) {
    m_upDla->DataUnregister(m_loadableIndex, m_vupOutputTensor[i]);
  }

  m_upDla->RemoveLoadable(m_loadableIndex);

  m_upDla.reset();

  for (auto i = 0u; i < m_pInputTensorScibuf.size(); i++) {
    if (m_pInputTensorScibuf[i]) {
      NvSciBufObjFree(m_pInputTensorScibuf[i]);
    }
  }

  for (auto i = 0u; i < m_vupInputTensor.size(); i++) {
    delete m_vupInputTensor[i];
    m_vupInputTensor[i] = nullptr;
  }

  for (auto i = 0u; i < m_pOutputTensorScibuf.size(); i++) {
    if (m_pOutputTensorScibuf[i]) {
      NvSciBufObjFree(m_pOutputTensorScibuf[i]);
    }
  }

  for (auto i = 0u; i < m_vupOutputTensor.size(); i++) {
    delete m_vupOutputTensor[i];
    m_vupOutputTensor[i] = nullptr;
  }

  DeinitNvSciBuf();
}

NvMediaStatus DlaTool::SetUp() {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvSciBufObj sciBufObj;

  if (m_isPingTest) {
    LOG_INFO("Ping test \n");
    goto fail;
  }

  status = InitNvSciBuf();
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "InitNvSciBuf");

  m_device = NvMediaDeviceCreate();
  PROPAGATE_ERROR_FAIL(m_device != nullptr, "NvMediaDeviceCreate");

  m_upDla = Dla::Create();
  PROPAGATE_ERROR_FAIL(m_upDla != nullptr, "Create");

  status = m_upDla->Init(m_dlaId, m_numTasks);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Init");

  status = m_upDla->AddLoadable(m_profileName, m_loadableIndex);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "AddLoadable");

  status = m_upDla->GetDesc(m_loadableIndex, m_InputTensorDesc, m_OutputTensorDesc);

  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetDesc");

  // hard-coded value for cuda device id
  // m_nvsci_manager = std::make_unique<NvSciManager>(m_NvscibufModule, this->m_upDla.get(), 0, cuda_stream_);

  // input tensor allocation
  for (auto i = 0u; i < m_InputTensorDesc.size(); i++) {
    status = ReconcileAndAllocSciBufObj(m_InputTensorDesc[i].tensorAttrs, m_InputTensorDesc[i].numAttrs, &sciBufObj);
    // status = ReconcileAndAllocSciBufObj(m_InputTensorDesc[i].tensorAttrs, m_InputTensorDesc[i].numAttrs,
    //                                     m_nvsci_manager->GetCudaNvSciSignaler()->getNvSciRawBufAttrList(), &sciBufObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "ReconcileAndAllocSciBufObj");

    m_pInputTensorScibuf.push_back(sciBufObj);  //

    Tensor* upTensor = new Tensor(m_device);

    status = upTensor->Create(sciBufObj);  // NvMediaTensorCreateFromNvSciBuf
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor Create");

    status = upTensor->SetData(0);  //
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor SetData");

    status = m_upDla->DataRegister(m_loadableIndex, upTensor);  //

    m_vupInputTensor.push_back(std::move(upTensor));
  }
  m_vupInputBufferGPU.resize(m_InputTensorDesc.size());
  for (auto i = 0u; i < m_InputTensorDesc.size(); i++) {
    cudaHostRegister(m_vupInputTensor[i]->GetTensorMapPtr(), m_vupInputTensor[i]->GetTensorMapSize(), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void**)&m_vupInputBufferGPU[i], (void*)m_vupInputTensor[i]->GetTensorMapPtr(), 0);
  }

  // output tensor allocation
  for (auto i = 0u; i < m_OutputTensorDesc.size(); i++) {
    status = ReconcileAndAllocSciBufObj(m_OutputTensorDesc[i].tensorAttrs, m_OutputTensorDesc[i].numAttrs, &sciBufObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "ReconcileAndAllocSciBufObj");

    m_pOutputTensorScibuf.push_back(sciBufObj);

    Tensor* upTensor = new Tensor(m_device);

    status = upTensor->Create(sciBufObj);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor Create");

    status = upTensor->SetData(0);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Tensor SetData");

    status = m_upDla->DataRegister(m_loadableIndex, upTensor);

    m_vupOutputTensor.push_back(std::move(upTensor));
  }
  m_vupOutputBufferGPU.resize(m_InputTensorDesc.size());
  for (auto i = 0u; i < m_OutputTensorDesc.size(); i++) {
    cudaHostRegister(m_vupOutputTensor[i]->GetTensorMapPtr(), m_vupOutputTensor[i]->GetTensorMapSize(), cudaHostRegisterMapped);
    cudaHostGetDevicePointer((void**)&m_vupOutputBufferGPU[i], (void*)m_vupOutputTensor[i]->GetTensorMapPtr(), 0);
  }

  LOG_DBG("m_vupInputTensor address after setUp:%p\n", m_vupInputTensor[0]->GetTensorMapPtr());
  LOG_DBG("m_vupOutputTensor address after setUp:%p\n", m_vupOutputTensor[0]->GetTensorMapPtr());

fail:
  return status;
}

NvMediaStatus DlaTool::Run(void* input_buffer, uint32_t input_size, int inputType, void* output_buffer,
                           uint32_t output_size, int outType) {
  // m_nvsci_manager->GetCudaNvSciSignaler()->signalExternalSemaphore();
  // LOG_DBG("GetCudaNvSciSignaler\n");
  // m_upDla->InsertPreSciFences(m_nvsci_manager->GetCudaSignalerFence());
  // LOG_DBG("InsertPreSciFences\n");
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  if (input_size > 0) {
    int inNum = m_vupInputTensor.size();
    if (inputType == 0)
      for (int i = 0; i < inNum; i++) {
        // only support models that only has one input if do not use direct I/O and cpu_input (inputType==0)
        if ((i == 0) && (inputType == 0)) {
          uint32_t cp_size = m_vupInputTensor[i]->GetTensorMapSize();
          if (cp_size > input_size) cp_size = input_size;
          m_vupInputTensor[i]->FillDataIntoTensor(cp_size, input_buffer);
        }

        LOG_DBG("m_vupInputTensor[%d] address :%p\n", i, m_vupInputTensor[i]->GetTensorMapPtr());
      }

    status = m_upDla->Submit(m_loadableIndex, m_vupInputTensor, m_vupOutputTensor);
    CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Submit");
  }

  if (output_size > 0) {
    int outNum = m_vupOutputTensor.size();
    uint32_t remain_size = output_size;
    char* ch_output_buffer = (char*)output_buffer;
    for (int i = 0; i < outNum; i++) {
      if (CheckStatus) {
        status = m_vupOutputTensor[i]->GetStatus();
        CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetStatus");
      }
      if (0 == outType) {
        uint32_t cp_size = m_vupOutputTensor[i]->GetTensorMapSize();
        if (cp_size > remain_size) cp_size = remain_size;
        if (cp_size > 0) {
          status = m_vupOutputTensor[i]->CopyDataFromTensor(cp_size, ch_output_buffer);
          if (ch_output_buffer) {  // copy data if ch_output_buffer != NULL
            ch_output_buffer += cp_size;
          }
          remain_size -= cp_size;
        }
      }

      LOG_DBG("m_vupOutputTensor[%d] address :%p\n", i, m_vupOutputTensor[i]->GetTensorMapPtr());
    }
  }

fail:
  return status;
}
/*
int DlaTool::CudaSignal() {
  m_nvsci_manager->GetCudaNvSciSignaler()->signalExternalSemaphore();

  return 0;
}
*/
void* DlaTool::GetOutPointer(int index) { return m_vupOutputTensor[index]->GetTensorMapPtr(); }
void* DlaTool::GetInputPointer(int index) { return m_vupInputTensor[index]->GetTensorMapPtr(); }

uint32_t DlaTool::GetOutSize(int index) { return m_vupOutputTensor[index]->GetTensorMapSize(); }

uint32_t DlaTool::GetInputSize(int index) { return m_vupInputTensor[index]->GetTensorMapSize(); }

static NvMediaStatus DecodeTensorDesc(NvMediaDlaTensorDescriptor* tensorDesc, EngineBuffer* pInfo) {
  uint32_t i = 0;
  LOG_DBG("Tensor descripor \n");
  LOG_DBG("\t name = %s \n", tensorDesc->name);

  // int nDataType = 4;
  int nBatchSize = 1;
  int nBitsPerElement = 16;
  int nDims = 0;  // < 8 = NVMEDIA_TENSOR_MAX_DIMENSIONS
  int dims[32];
  for (i = 0; i < tensorDesc->numAttrs; i++) {
    switch (tensorDesc->tensorAttrs[i].type) {
      case NVM_TENSOR_ATTR_DATA_TYPE:
        LOG_DBG("\t Data type = %d \n", tensorDesc->tensorAttrs[i].value);
        // nDataType = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_BITS_PER_ELEMENT:
        LOG_DBG("\t Bits per element = %d \n", tensorDesc->tensorAttrs[i].value);
        nBitsPerElement = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_DIMENSION_ORDER:
        LOG_DBG("\t dimension order = %d \n", tensorDesc->tensorAttrs[i].value);
        break;
      case NVM_TENSOR_ATTR_CPU_ACCESS:
        LOG_DBG("\t CPU access = %d \n", tensorDesc->tensorAttrs[i].value);
        break;
      case NVM_TENSOR_ATTR_ALLOC_TYPE:
        LOG_DBG("\t Alloc type = %d \n", tensorDesc->tensorAttrs[i].value);
        break;
      case NVM_TENSOR_ATTR_4D_N:
        LOG_DBG("\t N = %d \n", tensorDesc->tensorAttrs[i].value);
        nBatchSize = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_4D_C:
        LOG_DBG("\t C = %d \n", tensorDesc->tensorAttrs[i].value);
        dims[nDims++] = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_4D_H:
        LOG_DBG("\t H = %d \n", tensorDesc->tensorAttrs[i].value);
        dims[nDims++] = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_4D_W:
        LOG_DBG("\t W = %d \n", tensorDesc->tensorAttrs[i].value);
        dims[nDims++] = tensorDesc->tensorAttrs[i].value;
        break;
      case NVM_TENSOR_ATTR_4D_X:
        LOG_DBG("\t X = %d \n", tensorDesc->tensorAttrs[i].value);
        dims[nDims++] = tensorDesc->tensorAttrs[i].value;
        break;
      default:
        return NVMEDIA_STATUS_ERROR;
    }
  }

  pInfo->name = tensorDesc->name;
  switch (nBitsPerElement) {
    case NVM_TENSOR_ATTR_BITS_PER_ELEMENT_16:
      pInfo->nDataType = 1;      // FP16
      pInfo->nTensorFormat = 4;  // CHW16
      break;
    case NVM_TENSOR_ATTR_BITS_PER_ELEMENT_8:
      pInfo->nDataType = 2;      // INT8
      pInfo->nTensorFormat = 5;  // CHW32
      break;
    default:
      LOG_DBG("Unsupported data type with bits per element = %d\n", nBitsPerElement);
      break;
  }
  // pInfo->nDataType = (NVM_TENSOR_ATTR_DATA_TYPE_FLOAT == nDataType) ? 1 : 2;
  pInfo->nMaxBatch = nBatchSize;
  pInfo->nDims = nDims;
  std::copy_n(dims, nDims, pInfo->d);

  return NVMEDIA_STATUS_OK;
}

int DlaTool::GetBufferInfo(EngineBuffer bufferInfo[]) {
  int inNum = m_vupInputTensor.size();
  int outNum = m_vupOutputTensor.size();
  if (inNum + outNum > ONNXTRT_MAX_BUFFERNUM) return 0;

  int idx = 0;
  for (int i = 0; i < inNum; i++, idx++) {
    bufferInfo[idx].nBufferType = 0;  // 0: gpu_Input
    bufferInfo[idx].nBufferSize = m_vupInputTensor[i]->GetTensorMapSize();
    // change cpu_input to gpu_input
    bufferInfo[idx].p = m_vupInputBufferGPU[i]; // m_vupInputTensor[i]->GetTensorMapPtr();
    // bufferInfo[idx].p = m_nvsci_manager->GetCudaNvSciSignaler()->cudaImportNvSciTensorBuf(m_pInputTensorScibuf[i]);

    DecodeTensorDesc(&m_InputTensorDesc[i], &bufferInfo[idx]);
  }

  for (int i = 0; i < outNum; i++, idx++) {
    bufferInfo[idx].nBufferType = 1;  // 1: gpu_output,  11: cpu_Output
    bufferInfo[idx].p = m_vupOutputBufferGPU[i]; // m_vupOutputTensor[i]->GetTensorMapPtr();
    // bufferInfo[idx].p = m_nvsci_manager->GetCudaNvSciSignaler()->cudaImportNvSciTensorBuf(m_pOutputTensorScibuf[i]);
    bufferInfo[idx].nBufferSize = m_vupOutputTensor[i]->GetTensorMapSize();

    DecodeTensorDesc(&m_OutputTensorDesc[i], &bufferInfo[idx]);
  }

  return idx;
}

NvMediaStatus DlaTool::InitNvSciBuf(void) {
  NvSciError err = NvSciError_Success;
  NvMediaStatus status = NVMEDIA_STATUS_ERROR;

  err = NvSciBufModuleOpen(&m_NvscibufModule);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufModuleOpen");

  status = NvMediaTensorNvSciBufInit();
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "NvMediaTensorNvSciBufInit");

fail:
  return status;
}

void DlaTool::DeinitNvSciBuf() {
  NvSciBufModuleClose(m_NvscibufModule);

  NvMediaTensorNvSciBufDeinit();
}

NvMediaStatus DlaTool::ReconcileAndAllocSciBufObj(NvMediaTensorAttr tensorAttrs[], uint32_t numAttrs,
                                                  NvSciBufObj* sciBuf) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvSciError err = NvSciError_Success;
  NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
  NvSciBufAttrList unreconciled_attrlistTensor = NULL;
  NvSciBufAttrList reconciled_attrlist = NULL;
  NvSciBufAttrList conflictlist = NULL;

  NvSciBufAttrKeyValuePair attr_kvp = {NvSciBufGeneralAttrKey_RequiredPerm, &access_perm, sizeof(access_perm)};

  err = NvSciBufAttrListCreate(m_NvscibufModule, &unreconciled_attrlistTensor);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListCreate");

  err = NvSciBufAttrListSetAttrs(unreconciled_attrlistTensor, &attr_kvp, 1);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListSetAttrs");

  status = Tensor::FillNvSciBufTensorAttrs(m_device, tensorAttrs, numAttrs, unreconciled_attrlistTensor);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetNvSciBufTensorAttrs");

  err = NvSciBufAttrListReconcile(&unreconciled_attrlistTensor, 1, &reconciled_attrlist, &conflictlist);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

  err = NvSciBufObjAlloc(reconciled_attrlist, sciBuf);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

  if (unreconciled_attrlistTensor) {
    NvSciBufAttrListFree(unreconciled_attrlistTensor);
  }
  if (reconciled_attrlist) {
    NvSciBufAttrListFree(reconciled_attrlist);
  }
  if (conflictlist) {
    NvSciBufAttrListFree(conflictlist);
  }

fail:
  return status;
}

NvMediaStatus DlaTool::ReconcileAndAllocSciBufObj(NvMediaTensorAttr tensorAttrs[], uint32_t numAttrs,
                                                  NvSciBufAttrList attrList, NvSciBufObj* sciBuf) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvSciError err = NvSciError_Success;
  NvSciBufAttrList unreconciled_attrlistTensor = NULL;
  NvSciBufAttrList rawBufUnreconciledList[2];
  NvSciBufAttrList reconciled_attrlist = NULL;
  NvSciBufAttrList conflictlist = NULL;

  err = NvSciBufAttrListCreate(m_NvscibufModule, &unreconciled_attrlistTensor);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListCreate");

  status = Tensor::FillNvSciBufTensorAttrs(m_device, tensorAttrs, numAttrs, unreconciled_attrlistTensor);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "GetNvSciBufTensorAttrs");

  rawBufUnreconciledList[0] = unreconciled_attrlistTensor;
  rawBufUnreconciledList[1] = attrList;

  err = NvSciBufAttrListReconcile(rawBufUnreconciledList, 2, &reconciled_attrlist, &conflictlist);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

  err = NvSciBufObjAlloc(reconciled_attrlist, sciBuf);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciBufAttrListReconcile");

  if (unreconciled_attrlistTensor) {
    NvSciBufAttrListFree(unreconciled_attrlistTensor);
  }
  if (reconciled_attrlist) {
    NvSciBufAttrListFree(reconciled_attrlist);
  }
  if (conflictlist) {
    NvSciBufAttrListFree(conflictlist);
  }

fail:
  return status;
}