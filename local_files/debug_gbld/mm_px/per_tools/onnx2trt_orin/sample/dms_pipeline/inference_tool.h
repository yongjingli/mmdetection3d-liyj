#ifndef DLA_INFERENCE_TOOL_H_
#define DLA_INFERENCE_TOOL_H_

#include <dlfcn.h>
#include <string>
#include <vector>
#include "onnxtrt.h"
#define MAX_OUT_BUFFER 5
#define MAX_IN_BUFFER 1
#define MAX_NUM_ENGINE 4

enum ENGINE_STATUS { CREATED = 0, DESTROYED = 1, LIB_ERROR = 2, RUNFAILED = 3, RUNWELL = 4 };
class DLAInferTool {
 private:
  void *plib_{nullptr};
  std::string engine_paths_[MAX_NUM_ENGINE];

  int engine_ids_[MAX_NUM_ENGINE];

  PTCreateEngine CreatEngine_;
  PTDestoryEngine DestoryEngine_;
  PTRunEngine RunEngine_;
  PTGetBufferOfEngine GetBuffer_;
  EngineBuffer *bufferInfo_[MAX_NUM_ENGINE];
  int num_bindings_{2};

  void *pstream_;

 public:
  DLAInferTool(const std::string &dla_lib_path);
  ENGINE_STATUS CreatEngine(const std::string &engine_path, int engine_id);  // CreateEngien;
  void DestoryEngine(int engine_id);                                         // DestoryEngine
  ENGINE_STATUS inference(char *input_data, int engine_id, char *out_data);
  ENGINE_STATUS GetInOutBuffer(void **ppstream, int engine_id, void *input_buffer[], int &in_num, int *in_size,
                               void *output_buffer[], int &out_num, int *out_size, int &out_size_sum);

  ~DLAInferTool() {
    this->plib_ = nullptr;

    dlclose(this->plib_);
  }
};

#endif  // DLA_INFERENCE_TOOL_H_
