#include <string.h>
#include "YhatsFilter.h"

#define SMEMSIZE 49152

__global__ void yHatsFilterCHW(const float* inData, float* outData, float offset, 
                                int outNum, int inNum, int radius, int N, 
                                int chOffset, int width, int height) {
    int    inOffset  = blockIdx.y * inNum;
    int    outOffset = blockIdx.y * outNum;

    const float* threshold = inData     + inOffset;
    const float* curData   = threshold  + width + (blockIdx.x + chOffset) * width * height;  

    float* outIdx    = outData    + outOffset + width;
    float* outBuff   = outIdx     + N * width + blockIdx.x * width * N;

    float curThreTmp = threshold[threadIdx.x] + offset;
    int   curThre    = (int)(curThreTmp);
    //cliping threshold value.
    curThre = curThre <= 0 ? 0 : curThre;
    curThre = curThre >= height ? height : curThre;
    
    int cnt = -1;
    for(int j = curThre - radius; j <= curThre + radius; ++j) {
        cnt ++;
        if(j < 0 || j >= height) continue;
        int curPos = threadIdx.x + cnt * width;
        outBuff[curPos] = curData[threadIdx.x + j * width];

        if(blockIdx.x == 0)
            reinterpret_cast<int*>(outIdx)[curPos] = j;
    }

    if(blockIdx.x == 0) {
        float* curOut = outData + outOffset;
        curOut[threadIdx.x] = curThreTmp - offset;
    }
        
        
}

int YhatsFilterForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                        std::vector<int> inDims, int chOffset, float threOffset, cudaStream_t stream) {
    
    int channel = inDims[0];
    int height  = inDims[1];
    int width   = inDims[2];
    int radius  = inDims[3];

    float offset = threOffset;

    int N = radius * 2 + 1;
    int outNum = ((channel + 1) * N * width + width);
    int inNum  = ((channel + chOffset) * height * width + width);
    // int maxN   = SMEMSIZE / 4 / width / 2;

    // if(N > maxN) {
    //     DPRINTF(1, "Error: The maximum filtered number per channel per x-axis is %d > %d(shared memory %d)\n", N, maxN, SMEMSIZE);
    //     return -1;
    // }

    const float* inDataDev = reinterpret_cast<const float*>(inputs[0]);
    float*       outBuff   = reinterpret_cast<float*>      (outputs[0]);

    cudaMemsetAsync(outBuff, 0xCC, sizeof(float) * batchSize * outNum, stream);

    dim3 gridSize (channel, batchSize, 1);
    dim3 blockSize(width, 1, 1);

    yHatsFilterCHW<<<gridSize, blockSize, 0, stream>>>
            (inDataDev, outBuff, offset, outNum, inNum, radius, N, chOffset, width, height);

    return 0;
}




