/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "nvsci_manager.h"
#include "DlaTool.h"

#define checkNvSciErrors(call)                              \
  do {                                                      \
    NvSciError _status = call;                              \
    if (NvSciError_Success != _status) {                    \
      printf(                                               \
          "NVSCI call in file '%s' in line %i returned"     \
          " %d, expected %d\n",                             \
          __FILE__, __LINE__, _status, NvSciError_Success); \
      fflush(stdout);                                       \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
  } while (0)

NvMediaStatus NvSciManager::CreateSyncObjFromAttrList(NvSciSyncAttrList list1, NvSciSyncAttrList list2,
                                                      NvSciSyncObj* syncObj) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  NvSciError err;
  NvSciSyncAttrList unreconcileDlast[2] = {};
  NvSciSyncAttrList reconcileDlalist = nullptr;
  NvSciSyncAttrList newConflictList = nullptr;

  unreconcileDlast[0] = list1;
  unreconcileDlast[1] = list2;

  // Reconcile Signaler and Waiter NvSciSyncAttrList
  err = NvSciSyncAttrListReconcile(unreconcileDlast, 2, &reconcileDlalist, &newConflictList);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncAttrListReconcile");

  // Create NvSciSync object and get the syncObj
  err = NvSciSyncObjAlloc(reconcileDlalist, syncObj);
  PROPAGATE_ERROR_FAIL(err == NvSciError_Success, "NvSciSyncObjAlloc");

fail:
  FreeSyncAttrList(reconcileDlalist);
  FreeSyncAttrList(newConflictList);

  return status;
}

NvMediaStatus NvSciManager::SetupNvSciSync(Dla* m_upDla, CudaNvSciSignal* cudaNvSciSignaler) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;
  // cudaExternalResInterop cudaExtResObj;
  NvSciSyncAttrList signalerAttrList, waiterAttrList;

  signalerAttrList = cudaNvSciSignaler->getNvSciSyncAttrList();
  checkNvSciErrors(NvSciSyncAttrListCreate(syncModule_, &waiterAttrList));
  status = m_upDla->GetAttrList(syncModule_, waiterAttrList, NVMEDIA_WAITER);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Waiter GetAttrList");

  CreateSyncObjFromAttrList(signalerAttrList, waiterAttrList, &cudaSignalerSyncObj_);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "CreateSyncObjFromAttrList between GPU signaler and Dla waiter");

  // Import semaphore to Cuda
  cudaNvSciSignaler->cudaImportNvSciSemaphore(cudaSignalerSyncObj_);

  // Register Syncobj for Dla
  status = m_upDla->RegisterSyncObj(NVMEDIA_PRESYNCOBJ, cudaSignalerSyncObj_);
  CHECK_FAIL(status == NVMEDIA_STATUS_OK, "Dla RegisterSyncObj");

  // status = m_upDla->RegisterSyncObj(NVMEDIA_EOFSYNCOBJ, nvMediaSignalerSyncObj);
  // if (status == NVMEDIA_STATUS_OK) {
  //   std::cout << "Dla RegisterSyncObj";
  // }

  // // Set EOF for Dla
  // status = m_upDla->SetEOFSyncObj(nvMediaSignalerSyncObj);
  // if (status == NVMEDIA_STATUS_OK) {
  //   std::cout << "Dla SetEOFSyncObj";
  // }
fail:
  /* Free Attribute list objects */
  FreeSyncAttrList(signalerAttrList);
  FreeSyncAttrList(waiterAttrList);

  return status;
}

void NvSciManager::initNvSci() {
  checkNvSciErrors(NvSciSyncModuleOpen(&syncModule_));
  // DLA has opened buffModule, so i do not need to do this here
  // checkNvSciErrors(NvSciBufModuleOpen(&buffModule));
  m_fence = (NvSciSyncFence*)calloc(1, sizeof(NvSciSyncFence));
}

NvSciManager::NvSciManager(NvSciBufModule& buffModule, Dla* m_upDla, int cudaDeviceId, cudaStream_t stream) {
  NvMediaStatus status = NVMEDIA_STATUS_OK;

  initNvSci();
  cudaNvSciSignaler_ = std::make_unique<CudaNvSciSignal>(buffModule, syncModule_, cudaDeviceId, m_fence, stream);
  SetupNvSciSync(m_upDla, cudaNvSciSignaler_.get());
  if (status == NVMEDIA_STATUS_OK) {
    std::cout << "SetupNvSciSync" << std::endl;
  }
}