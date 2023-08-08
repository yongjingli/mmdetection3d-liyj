/*
 * Copyright (c) 2020, Xpeng Motor. All rights reserved.
 * libonnxtrt_dla.so : support NvMeadia DLA runtime.
 * 2020-5-8 : Create.
 */

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>

#include "DlaTool.h"
#include "ConfidenceFilterDLA.h"
#include "VoxelGenerator.h"
#include "onnxtrt.h"
#include "utils.h"

struct GpuTimer {
  void *start;
  void *stop;

  GpuTimer() {
    cudaEvent_t *p_start = (cudaEvent_t *)&start;
    cudaEvent_t *p_stop = (cudaEvent_t *)&stop;
    cudaEventCreate(p_start);
    cudaEventCreate(p_stop);
  }
  ~GpuTimer() { 
    cudaEventDestroy((cudaEvent_t)start);
    cudaEventDestroy((cudaEvent_t)stop);
  }

  void Start() { 
    cudaEventRecord((cudaEvent_t)start, 0); 
  }
  void Stop(){ 
    cudaEventRecord((cudaEvent_t)stop, 0);
  }
  float Elapsed() { 
    float elapsed;
    cudaEventSynchronize((cudaEvent_t)stop);
    cudaEventElapsedTime(&elapsed, (cudaEvent_t)start, (cudaEvent_t)stop);
    return elapsed;
  }
};


class IDLAWorkspace {
 private:
  int status = 0;  //-1, 0: not inited, >=1: inited,
  int ch = 0;
  DlaTool *dla_ptr = nullptr;

  // index for buffer, usefull when different device has different buffer order
  std::vector<int> bufferIndex;
  int nbBindings = 0;
  int nMaxBatchsize = 0;
  EngineBuffer bufferInfo[ONNXTRT_MAX_BUFFERNUM];
  cudaStream_t stream = nullptr;
  int frameID = 0;
  std::string savePath = "";  // save input data to path
  std::string loadFile = "";  // load input data from file

  FilterParam *d_params = nullptr;
  void *d_output = nullptr;
  int outbuf_size = 0; // Byte

  //lidar params
  VoxelGenerator* _voxelGen = nullptr;
  bool _isVgInitialzed = false;
  LidarFilterParam *d_lidar_params = nullptr;

  int priority = 0;

 public:
 //gpu timer
  GpuTimer* _gpuTimer = nullptr;
  bool  _isGtInitialized = false;
  float _timeAccumulator = 0.0f;
  int   _iterationTimes = 0;
  bool  _warmedUp = false;

  IDLAWorkspace() { memset(bufferInfo, 0, sizeof(bufferInfo)); };
  ~IDLAWorkspace() { release(); };

  void setStatus(int s) { status = s; }
  bool isInited() { return (nullptr != dla_ptr && status > 0); }
  int init(int setch, uint32_t numTasks, std::string loadableName, const char *config);

  void doInference(void *inputData, void *outData, int batchSize = 1, int inputType = 0, int outType = 0);

  void release();

  // get input/output buffer infomation ; get cudaStream ;
  int getBufferInfo(EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
    if (NULL == pNum || NULL == pBufferInfo) return ONNXTRT_PARAMETER_ERR;
    if (NULL == pStream)
      cudaStreamSynchronize(stream);
    else
      *pStream = stream;
    *pNum = nbBindings;
    *pBufferInfo = bufferInfo;
    return 0;
  }

  int saveBuf(const char* magic, void *buf, int type, int64_t size) {
    if (buf == nullptr || size <= 0) return -1;

    char *cpuBuf = (char*)buf;
    if( 1 == type ){ // GPU buffer
      cpuBuf = new char[size];
      cudaMemcpyAsync(cpuBuf, buf, size, cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
          
    char outfile[260];
    snprintf(outfile, 256, "%s/%s_ch%d_%d.bin", savePath.c_str(), magic, ch, frameID);
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file.write(cpuBuf, size);
    file.close();
    if( cpuBuf != buf){ 
      delete[] cpuBuf;
    }

    DPRINTF(2, "Save DLA %s buffer %s size=%ld\n", magic, outfile, size);
    return size;
  }  
};

int IDLAWorkspace::init(int setch, uint32_t numTasks, std::string loadableName, const char *config) {
  bool testPing = false;
  int dlaId = setch % 2;
  if((nullptr != config) && strstr(config, "DLA=")) { // get config from config string 
    if( strstr( config, "DLA=1" ) ){
      dlaId = 1;
    } else {
      dlaId = 0;
    } 
  }
	DPRINTF(1, "[CH%d] dlaId = %d\n", ch, dlaId);

  { // get config from env
    char *val = getenv("DLA_SAVEPATH");
    if (NULL != val) {
      savePath = val;
      DPRINTF(2, "DLA_SAVEPATH = %s\n", savePath.c_str());
    }
    val = getenv("DLA_LOADFILE");
    if (NULL != val) {
      loadFile = val;
      DPRINTF(2, "DLA_LOADFILE = %s\n", loadFile.c_str());
    }
  }

  if (nullptr != config) {  // get config from config string
    priority = (ch == 0) ? -1 : 0;
    if (strstr(config, "Priority=")) {
      if (strstr(config, "Priority=High")) {
        priority = -1;
      } else {
        priority = 0;
      }
      DPRINTF(1, "[CH%d] Priority = %d\n", ch, priority);
    }
    // infer->setDLACore(ch % 2);  // DLA function moved to libonnxtrt_dla.so
    // DPRINTF(2, "[CH%d] infer->getDLACore=%d\n", ch, infer->getDLACore());
  }

	cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);

  dla_ptr = new DlaTool(dlaId, numTasks, loadableName, testPing, stream);
  NvMediaStatus nvmstatus = dla_ptr->SetUp();
  if (nvmstatus != NVMEDIA_STATUS_OK) {
    LOG_ERR("DLA setup fails \n");
    return ONNXTRT_PARAMETER_ERR;
  }

  nbBindings = dla_ptr->GetBufferInfo(bufferInfo);
  nMaxBatchsize = bufferInfo[0].nMaxBatch;

  {

    char *measureTimeChar = getenv("TRT_MEASURETIME"); 
    int measureTime = 0;
    if(NULL != measureTimeChar)
      measureTime = atoi(measureTimeChar);
    
    DPRINTF(2, "initializeing gpu timer\n");
    if(measureTime == 1) {
      _gpuTimer = new GpuTimer;
      _isGtInitialized = true;
    }

    DPRINTF(2, "gpu timer is initialized\n");
  }

  DPRINTF(2,"The nbinding is: %d\n", nbBindings);
  DPRINTF(2,"The input name is: %s\n", bufferInfo[0].name); 
  DPRINTF(2,"The output name is: %s\n", bufferInfo[1].name);
  _isVgInitialzed = false;
  if(NULL != strcasestr(bufferInfo[0].name, "lidar_pcd")) {
    if(0 == memcmp(bufferInfo[0].name, "/", 1)){
      std::vector<float> pcRangeUpper(3, 0.0f);
      std::vector<float> pcRangeLower(3, 0.0f);
      std::vector<float> resolution  (3, 0.0f);
      float scale;
      // const char *param_str = bufferInfo[0].name + 3;
      const char *param_str = bufferInfo[0].name;
      // vg:/R_xx_xx_xx/X_xx_xx/Y_xx_xx/Z_xx_xx/S_xx
      sscanf(param_str, "/%f_%f/%f_%f/%f_%f/%f_%f/%f", &resolution[0], 
            &resolution[2], &pcRangeLower[0], &pcRangeUpper[0],
            &pcRangeLower[1], &pcRangeUpper[1], &pcRangeLower[2], &pcRangeUpper[2],
            &scale);
      
      resolution[1] = resolution[0];

      _voxelGen = new VoxelGenerator(pcRangeUpper, pcRangeLower, resolution, 
                                     stream, bufferInfo[0].nDataType, scale);

      DPRINTF(2, "vg params: /R_%f_%f_%f/X_%f_%f/Y_%f_%f/Z_%f_%f/S_%f\n", resolution[0], 
            resolution  [1], resolution  [2], pcRangeLower[0], pcRangeUpper[0],
            pcRangeLower[1], pcRangeUpper[1], pcRangeLower[2], pcRangeUpper[2],
            scale);

    }else {
      _voxelGen = new VoxelGenerator(stream, bufferInfo[0].nDataType, 255.0f);
    }
    _voxelGen->initWorkSpace();
    _isVgInitialzed = true;

    bufferInfo[0].name = "lidar_pcd";
  }



  DPRINTF(2, "the input size is %d: %d -- %d -- %d.\n", bufferInfo[0].nBufferSize,
         bufferInfo[0].d[0], bufferInfo[0].d[1], bufferInfo[0].d[2]);

  DPRINTF(2, "the output size is %d: %d -- %d -- %d.\n", bufferInfo[1].nBufferSize,
         bufferInfo[1].d[0], bufferInfo[1].d[1], bufferInfo[1].d[2]);

  DPRINTF(2, "The tensor formate is %d\n", bufferInfo[1].nTensorFormat);

  // if(0 == memcmp(bufferInfo[0].name, "lidar_pcd", 9)) {
  //   _voxelGen = new VoxelGenerator(stream, bufferInfo[0].nDataType);
  //   _voxelGen->initWorkSpace();
  //   _isVgInitialzed = true;
  // }

  //First Get Filter Info from Output Name
  // "tfl2_filter/M33/N64/C0/T0.5/S16_237_457/tfl2" : M: mode, N: max_num, C: conf_offset, T: threshold, S: fpn_shape
  if (0 == memcmp(bufferInfo[1].name, "tfl2_filter", 11)) {  // strlen(name) <= 80
    cudaHostAlloc(&d_params, sizeof(FilterParam), cudaHostAllocMapped);
    const char *param_str = bufferInfo[1].name + 11;
    sscanf(param_str, "/M%d/N%d/C%d/T%f/S%d_%d_%d/O_%x_%x_%x", &d_params->mode, &d_params->max_num, d_params->_conf_offset,
           d_params->thresholds, &d_params->fpn_shape[0], &d_params->fpn_shape[1], &d_params->fpn_shape[2],
           (int*)&d_params->out_scale[0], (int*)&d_params->out_scale[1], (int*)&d_params->out_scale[2]);
    DPRINTF(2, "Filter: M%d/N%d/C%d/T%.3f/S%d_%d_%d/O_%.2f_%.2f_%.2f\n", d_params->mode, d_params->max_num, d_params->_conf_offset[0],
            d_params->thresholds[0], d_params->fpn_shape[0], d_params->fpn_shape[1], d_params->fpn_shape[2],
            d_params->out_scale[0], d_params->out_scale[1], d_params->out_scale[2]);

    int _ch = d_params->fpn_shape[0];
    d_params->input_num = nbBindings - 1;
    d_params->data_type = bufferInfo[1].nDataType;
    for (int i = 0; i < d_params->input_num; i++) {
      auto d_input_dim = &d_params->_input_dims[i];
      d_input_dim->x = bufferInfo[i + 1].d[2];
      d_input_dim->y = bufferInfo[i + 1].d[1];
      {
        d_input_dim->z = sqrtf(bufferInfo[i + 1].d[0] / _ch);  // ratio
        d_input_dim->w = d_input_dim->x * d_input_dim->z * d_input_dim->y * d_input_dim->z;
      }
      d_params->_input_size[i] = bufferInfo[i + 1].d[0] * bufferInfo[i + 1].d[1] * bufferInfo[i + 1].d[2];
      d_params->_d_input_bufs[i] = (const float *)bufferInfo[i + 1].p;

      DPRINTF(2, "FPN[%d] in(%d,%d,%d) -> DLA(%d,%d,%d,%d)\n", i, bufferInfo[i + 1].d[0], bufferInfo[i + 1].d[1],
              bufferInfo[i + 1].d[2], d_input_dim->x, d_input_dim->y, d_input_dim->z, d_input_dim->w);
    }

    nbBindings = 2;
    //bufferInfo[0].name = "image_pad";
    //bufferInfo[0].d[1] = 237;
    //bufferInfo[0].d[2] = 457;
        
    bufferInfo[1].name = "tfl2_filter";
    bufferInfo[1].nBufferType = 11;  // cpu_output
    bufferInfo[1].nDataType = 0;
    bufferInfo[1].nTensorFormat = 0;
    bufferInfo[1].nDims = 2; 
    bufferInfo[1].d[0] = d_params->max_num;
    bufferInfo[1].d[1] = _ch + 1;
    bufferInfo[1].d[2] = 0;
    outbuf_size = bufferInfo[1].d[0] * bufferInfo[1].d[1] * sizeof(float);
    cudaMallocHost(&d_output, outbuf_size);

    bufferInfo[1].p = d_output;
    
  } else if(NULL != strcasestr(bufferInfo[1].name, "lidar_mod_filter")) {
    
    DPRINTF(2, "Doing Lidar Filter Init routine\n");

    d_lidar_params = new LidarFilterParam;
    const char *param_str = bufferInfo[1].name;
    // "/M01/N64/C0/T0.5/S27_128_128/O0.120" : M: mode, N: max_num, C: conf_offset, T: threshold, S: fpn_shape O: int8 Scale
    sscanf(param_str, "/%d/%d/%d/%f/%d_%d_%d/%f", &d_lidar_params->mode, 
           &d_lidar_params->max_num, &d_lidar_params->_conf_offset[0],
           &d_lidar_params->thresholds[0], &d_lidar_params->fpn_shape[0],
           &d_lidar_params->fpn_shape[1],  &d_lidar_params->fpn_shape[2],
           &d_lidar_params->out_scale[0]);

    DPRINTF(2, "Filter: /M%d/N%d/C%d/T%.3f/S%d_%d_%d/O%.4f\n", d_lidar_params->mode, 
           d_lidar_params->max_num, d_lidar_params->_conf_offset[0],
           d_lidar_params->thresholds[0], d_lidar_params->fpn_shape[0],
           d_lidar_params->fpn_shape[1], d_lidar_params->fpn_shape[2],
           d_lidar_params->out_scale[0]);

    d_lidar_params->input_num = nbBindings - 1;
    d_lidar_params->data_type = bufferInfo[1].nDataType;
    for (int i = 0; i < d_lidar_params->input_num; i++) {
      // d_lidar_params->_input_size[i] = bufferInfo[i + 1].d[0] * bufferInfo[i + 1].d[1] * bufferInfo[i + 1].d[2];
      d_lidar_params->_input_size[i] = d_lidar_params->fpn_shape[0] * d_lidar_params->fpn_shape[1] *d_lidar_params->fpn_shape[2];
      d_lidar_params->_d_input_bufs[i] = (const void *)bufferInfo[i + 1].p;

      // DPRINTF(2, "FPN[%d] in(%d,%d,%d)\n", i, bufferInfo[i + 1].d[0], bufferInfo[i + 1].d[1], bufferInfo[i + 1].d[2]);
    }


    
    bufferInfo[1].name = "lidar_mod_filter";
    bufferInfo[1].nBufferType = 11;  // cpu_output
    bufferInfo[1].nDataType = 0;
    bufferInfo[1].nTensorFormat = 0;
    bufferInfo[1].nDims = 2; 
    bufferInfo[1].d[0] = d_lidar_params->max_num;
    bufferInfo[1].d[1] = d_lidar_params->fpn_shape[0] + 1;
    bufferInfo[1].d[2] = 0;
    outbuf_size = bufferInfo[1].d[0] * bufferInfo[1].d[1] * sizeof(float);
    bufferInfo[1].nBufferSize = outbuf_size;

    // bufferInfo[1].nDims = 3; 
    // outbuf_size = bufferInfo[1].d[0] * bufferInfo[1].d[1] * bufferInfo[1].d[2] * sizeof(float);
    // bufferInfo[1].nBufferSize = outbuf_size;

    // cudaMallocHost(&d_output, outbuf_size);
    cudaMalloc(&d_output, outbuf_size);

    //Allocate counter space
    cudaMalloc(&(d_lidar_params->_count[0]), sizeof(int));

    bufferInfo[1].p = d_output;
    DPRINTF(2, "Lidar Filter Init Finish\n");
  }
  else {
    for (int i = 1; i < nbBindings; i++) {
      if ( 1 == bufferInfo[i].nBufferType ) { //gpu output
        bufferInfo[i].nBufferType = 11;  // cpu_output
        bufferInfo[i].p = dla_ptr->GetOutPointer(i - 1);
      }
    }
  }
  
  return 0;
}

void writeResultBin(float *opData, int writeNum) {
    // const int imageSize = imageWidth * imageHeight;
  
    unsigned int valNum = static_cast<unsigned int>(writeNum);
    std::string outFileName ="./test.bin";
    std::ofstream out(outFileName, std::ios_base::binary);
    if(out.good()) {
        out.write((const char*)opData, valNum *  sizeof(float));
        out.close();
    }else {
        cout<<"Error: Cannot write to binary file. Return."<<endl;
    }
}

void IDLAWorkspace::doInference(void *inputData, void *outData, int batchSize, int inputType, int outType) {
  DPRINTF(2, "DLA inType=%d outType=%d\n", inputType, outType);

  if (nbBindings < 2) return;
  // Get input size. if inputType < 0 , no input data.
  int inSize = (inputType >= 0) ? bufferInfo[0].nBufferSize : 0;
  if( inSize > 0 ){ // copy GPU data to Tensor
    frameID++;
    
    if(1 == inputType){
      if (inputData != NULL && bufferInfo[0].p != inputData) {
        cudaMemcpyAsync(bufferInfo[0].p, inputData, inSize, cudaMemcpyDeviceToDevice, stream);
        if (!savePath.empty()) 
          saveBuf("gpu", bufferInfo[0].p, inputType, inSize);        
      }
    }else if (4 == inputType) {
      if(!_isVgInitialzed) {
        printf("Error: The input format is pcd, but VG is not initialized. Retrun.\n");
        return;
      }
      DPRINTF(2, "start generating voxels\n");
      LidarInput *lidarInput = reinterpret_cast<LidarInput*>(inputData);
      void* pointCloud = lidarInput->pointCloud;
      _voxelGen -> copyData(reinterpret_cast<PointXYZI*>(pointCloud), lidarInput->pointNum);
      _voxelGen -> generateVoxels(bufferInfo[0].p); 

      DPRINTF(2, "Padding voxel num is %d.\n", _voxelGen -> getTotVoxelNum());
      DPRINTF(2, "Real voxel num is %d.\n", _voxelGen -> getRealVoxelNum());

      // int totalVoxelNum = _voxelGen -> getTotVoxelNum();
 
      // uint16_t *tmpTest = new uint16_t[totalVoxelNum];
      // cudaMemcpyAsync(tmpTest, bufferInfo[0].p, totalVoxelNum * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);

      // int oneCnt  = 0;
      // int inValidCnt = 0;
      // for(int iter = 0; iter < totalVoxelNum; ++iter) {
      //   uint16_t tmpValue = tmpTest[iter];
      //   if(tmpValue != 0)
      //     oneCnt ++;
      //   else 
      //     inValidCnt ++;
      // }

      // cout<<"The one cnt is "<<oneCnt<<"; invalid cnt is "<<inValidCnt<<endl;

      // delete[] tmpTest;

      DPRINTF(2, "voxel generation finish\n");
      
    } else{
      DPRINTF(2, "using CPU data directly, the input size is %d\n", bufferInfo[0].nBufferSize);
      // inputData = new char[bufferInfo[0].nBufferSize];
      // memset(inputData, 0, bufferInfo[0].nBufferSize);
      if (!savePath.empty()) saveBuf("cpu", inputData, inputType, inSize);
      bufferInfo[0].p = inputData;
      // bufferInfo[0].p = nullptr;
    }
  } 
     
  // Support multi output
  int outSize = 0;
  int dla_outType = outType;
  if( 0 == dla_outType && (d_params || d_lidar_params) ) {
    dla_outType = 3; // optimizied for Filter in DLA model
  }
  if (outType >= 0 ) {
    for (auto bufferInfo_ : bufferInfo) {
      if (11 == bufferInfo_.nBufferType || 1 == bufferInfo->nBufferType) {  // cpu output or gpu output
        outSize += bufferInfo_.nBufferSize;
      }
    }
  }

  
  // int outSize = (outType >= 0) ? bufferInfo[1].nBufferSize : 0;
  NvMediaStatus nvmstatus = dla_ptr->Run(inputData, inSize, inputType, outData, outSize, dla_outType);
  if (nvmstatus != NVMEDIA_STATUS_OK) {
    LOG_ERR("DLA Run fails \n");
    // status = 2;
  }
  DPRINTF(2, "DLA Run Finish \n");
  if (outData && (d_params || d_lidar_params)) {

    DPRINTF(2, "Doing DLA Filter Routine \n");
    bool direct_out = (2 == abs(outType)); // dircet use outData, Pinned memory.
    if(d_params) {
      if (!direct_out) {
        d_params->outputs[0] = d_output;
      } else {
        d_params->outputs[0] = outData;
      }

      ConfidenceFilterForDLA(batchSize, d_params, stream);
      DPRINTF(3, "DLA Run ConfidenceFilter \n");
    } else if(d_lidar_params) {
      if (!direct_out) {
         DPRINTF(2, "No Direct Out in DLA \n");
        d_lidar_params->outputs[0] = d_output;
      } else {
        d_lidar_params->outputs[0] = outData;
      }

      ConfidenceFilterForLidarDLA(d_lidar_params, stream);
      DPRINTF(3, "DLA Run ConfidenceFilter for Lidar \n");
    }
  
    if (!direct_out) {
      cudaMemcpyAsync(outData, d_output, outbuf_size, cudaMemcpyDeviceToHost, stream);
    }
    
    if( outType >= 0 ){
      cudaStreamSynchronize(stream);
    }
    
    if (!savePath.empty()) saveBuf("out", d_output, 0, outbuf_size);
  }
}

void IDLAWorkspace::release() {
  status = -2;
  nbBindings = 0;
  if (nullptr != dla_ptr) {
    delete dla_ptr;
    dla_ptr = nullptr;
  }
  if (nullptr != d_params) {
    cudaFreeHost(d_params);
    d_params = nullptr;
    cudaFreeHost(d_output);
    d_output = nullptr;
  }

  if(nullptr != d_lidar_params) {
    cudaFree(d_lidar_params->_count[0]);
    cudaFree(d_output);
    delete d_lidar_params;
    d_lidar_params = nullptr;
  }


  if (nullptr != _voxelGen) {
    _voxelGen -> terminate();
    delete _voxelGen;
    _voxelGen = nullptr;
  }

  if (nullptr != _gpuTimer) {
    printf("The latency of inference is %.5lf ms, with %d iterations.\n", _timeAccumulator, _iterationTimes);
    printf("The average speed of inference is %.5lf ms\n", _timeAccumulator/_iterationTimes);
    delete _gpuTimer;
    _gpuTimer = nullptr;
  }

  status = 0;
}

const int MAX_BATCH = ONNXTRT_MAX_BATCHNUM;
const int MAX_CH = ONNXTRT_MAX_ENGINENUM;
static IDLAWorkspace gworkspace[MAX_CH];

// interface for so/python
/*
 * ch: channel ID for multiple models:  0:maskrcnn, 1:resnet ...
 * engine_filename: tensorrt engine file
 * pMaskWeight: mask weight for maskrcnn/retinamask
 */
extern "C" int CreateEngine(int ch, const char *engine_filename, const char *config_string) {
  if (ch >= 0 && ch < ONNXTRT_MAX_ENGINENUM) {
    if (gworkspace[ch].isInited()) return -2;
  } else if (-1 == ch) {  //-1: Find available ID
    for (int i = ONNXTRT_MAX_ENGINENUM - 1; i >= 0; i--) {
      if (!gworkspace[i].isInited()) {
        ch = i;
        break;
      }
    }
    if (-1 == ch) return -2;
  } else
    return ONNXTRT_PARAMETER_ERR;

  gworkspace[ch].init(ch, ONNXTRT_MAX_ENGINENUM - 1, engine_filename, config_string);
  gworkspace[ch].setStatus(1);

  return ch;
}

// Supported multiply input and output Tensor on DLA.
// But only copy one buffer throw interface "RunEngine".
// Please Get multi-IO pointers throw GetBufferOfEngine to avoid memcpy.
// inputType: 0:cpu buffer, 1:gpu buffer,<0: avoid memcpy and inference
// outType: 0:cpu buffer, 1:gpu buffer,<0: avoid memcpy and sync
extern "C" int RunEngine(int ch, int batch, char *inputData, int inputType, char *outData, int outType) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  if(gworkspace[ch]._isGtInitialized && gworkspace[ch]._warmedUp) {
    gworkspace[ch]._gpuTimer->Start();
    gworkspace[ch].doInference(inputData, outData, batch, inputType, outType);
    gworkspace[ch]._gpuTimer->Stop();
    if(outType >= 0) gworkspace[ch]._iterationTimes  += 1;
    gworkspace[ch]._timeAccumulator += gworkspace[ch]._gpuTimer->Elapsed();
    
  } else {
    gworkspace[ch].doInference(inputData, outData, batch, inputType, outType);
    gworkspace[ch]._warmedUp = true;
  }
    
  return ONNXTRT_OK;
}

extern "C" int DestoryEngine(int ch) {
  if (!gworkspace[ch].isInited()) {
    return -2;
  }
  gworkspace[ch].release();

  return ONNXTRT_OK;
}

extern "C" int GetBufferOfEngine(int ch, EngineBuffer **pBufferInfo, int *pNum, void **pStream) {
  return gworkspace[ch].getBufferInfo(pBufferInfo, pNum, pStream);
}

int TRT_DEBUGLEVEL = 1;
// static funtion called when using dlopen & dlclose
static __attribute__((constructor)) void lib_init(void) {
  {
    char *val = getenv("TRT_DEBUGLEVEL");
    if (NULL != val) {
      TRT_DEBUGLEVEL = atoi(val);
      SET_LOG_LEVEL((CLogger::LogLevel)TRT_DEBUGLEVEL);
    }
  }
  printf("Load onnx2trt DLA lib %s built@%s %s DebugLevel=%d\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__,
         TRT_DEBUGLEVEL);
}

static __attribute__((destructor)) void lib_deinit(void) {
  printf("Unload onnx2trt DLA lib %s built@%s %s\n", ONNXTRT_VERSION_STRING, __DATE__, __TIME__);
}
