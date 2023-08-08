#ifndef _DATA_CONVERT_H_
#define _DATA_CONVERT_H_

#include "onnxtrt.h"

void TransFormat(char const* inDataPtr, int batchSize, int ch, int inDataType, 
                 EngineBuffer *bufferInfo, cudaStream_t stream);

#endif