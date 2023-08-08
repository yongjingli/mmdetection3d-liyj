#include "inference_tool.h"
#include <iostream>
DLAInferTool::DLAInferTool(const std::string &dla_lib_path) {
  this->plib_ = dlopen(dla_lib_path.c_str(), RTLD_LAZY);
  if (this->plib_ == nullptr) {
    // printf("ERROR:%s:dlopen\n", dlerror());
    std::cerr << "ERROR dlopen failed !!!" << std::endl;
  }

  this->CreatEngine_ = (PTCreateEngine)dlsym(plib_, "CreateEngine");
  this->RunEngine_ = (PTRunEngine)dlsym(plib_, "RunEngine");
  this->DestoryEngine_ = (PTDestoryEngine)dlsym(plib_, "DestoryEngine");
  this->GetBuffer_ = (PTGetBufferOfEngine)dlsym(plib_, "GetBufferOfEngine");
  if (CreatEngine_ == nullptr || RunEngine_ == nullptr || DestoryEngine_ == nullptr || GetBuffer_ == nullptr) {
    std::cerr << "ERROR get function in dla_lib failed !!!" << std::endl;
  }
}
ENGINE_STATUS DLAInferTool::CreatEngine(const std::string &engine_path, int engine_id) {
  if (this->CreatEngine_(engine_id, engine_path.c_str(), nullptr) >= 0) {
    return CREATED;
  } else {
    return DESTROYED;
  }
}

void DLAInferTool::DestoryEngine(int engine_id) { this->DestoryEngine_(engine_id); }

ENGINE_STATUS DLAInferTool::GetInOutBuffer(void **ppStream, int engine_id, void *input_buffer[], int &in_num,
                                           int *in_size, void *output_buffer[], int &out_num, int *out_size,
                                           int &out_size_sum) {
  if (this->GetBuffer_(engine_id, &bufferInfo_[engine_id], &num_bindings_, ppStream) != 0) {
    return RUNFAILED;
  }

  in_num = 0;
  out_num = 0;
  out_size_sum = 0;
  for (int i = 0; i < num_bindings_; ++i) {
    if (bufferInfo_[engine_id][i].nBufferType == 0) {
      input_buffer[in_num] = bufferInfo_[engine_id][i].p;
      in_size[in_num] = bufferInfo_[engine_id][i].nBufferSize;
      in_num++;
    } else {
      output_buffer[out_num] = bufferInfo_[engine_id][i].p;
      out_size[out_num] = bufferInfo_[engine_id][i].nBufferSize;
      out_size_sum += bufferInfo_[engine_id][i].nBufferSize;
      out_num++;
    }
  }
  return RUNWELL;
}

ENGINE_STATUS DLAInferTool::inference(char *input_data, int engine_id, char *out_data) {
  // the input is on gpu,so set input type to 1
  if (this->RunEngine_(engine_id, 1, input_data, 1, out_data, -1) >= 0)  // inference
  {
    if (this->RunEngine_(engine_id, 1, nullptr, -1, out_data, 0) >= 0)  // Sync
    {
      return RUNWELL;
    }
  }
  return RUNFAILED;
}
