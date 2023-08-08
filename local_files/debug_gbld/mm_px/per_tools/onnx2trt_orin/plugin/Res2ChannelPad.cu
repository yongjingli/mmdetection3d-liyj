#include <iostream>
#include "Res2ChannelPad.h"

using namespace std;

__global__ void Res2ChannelOptPaddingUint8(const uint8_t * src, int8_t* dst,
                                            const uint32_t iWidth, const uint32_t iHeight,
                                            const uint32_t hWidth, const uint32_t hHeight,
                                            const uint32_t rHeight,const uint32_t channel,
                                            const uint32_t nWidth, const uint32_t nHeight,
                                            const uint32_t iOffset, const uint32_t oOffset) {
    extern uint8_t __shared__ smemBufferI[];

    const uint8_t* batchSrc = src + blockIdx.z * iOffset;
    int8_t*        batchDst = dst + blockIdx.z * oOffset;
                                               
    unsigned int readNumber = rHeight * iWidth;
    unsigned int startPos   = blockIdx.y * iWidth * iHeight + blockIdx.x * readNumber;

    for(unsigned int i = threadIdx.x; i < readNumber; i += blockDim.x) {
        uint8_t val = batchSrc[startPos + i];
        unsigned int subH    = i / iWidth;
        unsigned int subC    = (i & 1) * 2 + (subH & 1);
        unsigned int subW    = i % iWidth;
        unsigned int smemPos = subC * hWidth + subW / 2;

        smemBufferI[smemPos] = val;
    }

    __syncthreads();

    for(unsigned int i = threadIdx.x; i < readNumber; i += blockDim.x) {

        uint8_t val = smemBufferI[i];

        unsigned int rowInSmem  = i / hWidth;
        unsigned int dstChannel = rowInSmem * channel + blockIdx.y;
        unsigned int dstWidth   = i % hWidth;
        unsigned int dstHeight  = (blockIdx.x * rHeight + (rowInSmem & 1)) / 2;
        
        
        unsigned int dstPos = dstChannel * nWidth * nHeight + dstHeight * nWidth + dstWidth;

        batchDst[dstPos] = (val / 2);

    }
}

__global__ void Res2ChannelOptPaddingFloat(const float * src, float* dst,
                                            const uint32_t iWidth, const uint32_t iHeight,
                                            const uint32_t hWidth, const uint32_t hHeight,
                                            const uint32_t rHeight,const uint32_t channel,
                                            const uint32_t nWidth, const uint32_t nHeight,
                                            const uint32_t iOffset, const uint32_t oOffset) {
    extern float __shared__ smemBufferF[];

    const float* batchSrc = src + blockIdx.z * iOffset;
    float*       batchDst = dst + blockIdx.z * oOffset;
                                               
    unsigned int readNumber = rHeight * iWidth;
    unsigned int startPos   = blockIdx.y * iWidth * iHeight + blockIdx.x * readNumber;

    for(unsigned int i = threadIdx.x; i < readNumber; i += blockDim.x) {
        float val = batchSrc[startPos + i];
        unsigned int subH    = i / iWidth;
        unsigned int subC    = (i & 1) * 2 + (subH & 1);
        unsigned int subW    = i % iWidth;
        unsigned int smemPos = subC * hWidth + subW / 2;

        smemBufferF[smemPos] = val;
    }

    __syncthreads();

    for(unsigned int i = threadIdx.x; i < readNumber; i += blockDim.x) {

        float val = smemBufferF[i];

        unsigned int rowInSmem  = i / hWidth;
        unsigned int dstChannel = rowInSmem * channel + blockIdx.y;
        unsigned int dstWidth   = i % hWidth;
        unsigned int dstHeight  = (blockIdx.x * rHeight + (rowInSmem & 1)) / 2;
        
        
        unsigned int dstPos = dstChannel * nWidth * nHeight + dstHeight * nWidth + dstWidth;

        batchDst[dstPos] = (val / 255.0f);
    }
}

int Res2ChannelPadForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                            std::vector<uint32_t> inDims, uint32_t rowPadding, uint32_t columnPadding,  
                            DataType inDataType, cudaStream_t stream) {
    
    uint32_t resolution  = inDims[3];
    uint32_t channel     = inDims[0];
    uint32_t imageHeight = inDims[1];
    uint32_t imageWidth  = inDims[2];

    uint32_t readHeight  = 2;

    uint32_t newChannel = channel * resolution * resolution;
    uint32_t newHeight  = imageHeight / resolution + rowPadding;
    uint32_t newWidth   = imageWidth  / resolution + columnPadding;

    uint32_t inOffset  = channel    * imageHeight * imageWidth;
    uint32_t outOffset = newChannel * newHeight   * newWidth;

    dim3 GridSize (imageHeight / readHeight, channel, batchSize);
    dim3 BlockSize(256, 1, 1);

    // inDataType = DataType::kINT8;
    // cout<<"The input data type is "<<(int)inDataType<<endl;
    // cout<<"The new channel is "<<newChannel<<endl;
    // cout<<"The new height is " <<newHeight<<endl;
    // cout<<"The new width is "<<newWidth<<endl;

    // uint8_t* hIn  = new uint8_t[inOffset];
    // int8_t* hOut  = new int8_t[outOffset];
    

    if(inDataType == DataType::kINT8) {
        // cout<<"Doing INT8 Rountine"<<endl;
        const uint8_t* inData    = reinterpret_cast<const uint8_t*>(inputs[0]);
        int8_t*        outBuffer = reinterpret_cast<int8_t*>       (outputs[0]);

        size_t smemSize = sizeof(uint8_t) * imageWidth * readHeight;
        if(smemSize >= 49152) {
            cout<<"Error: Shared memory exceed the limit in Res2ChannelPadd"<<endl;
            return - 1;
        }

        // cudaMemcpy(hIn, inData, sizeof(int8_t) * inOffset, cudaMemcpyDeviceToHost);

            
        Res2ChannelOptPaddingUint8<<<GridSize, BlockSize, sizeof(uint8_t) * imageWidth * readHeight>>>
                                    (inData, outBuffer, imageWidth, imageHeight, imageWidth / 2, imageHeight / 2,
                                    readHeight, channel, newWidth, newHeight, inOffset, outOffset);

        // cudaMemcpy(hOut, outBuffer, sizeof(int8_t) * outOffset, cudaMemcpyDeviceToHost);


        // for(int i = 0; i < 64; ++i)
        //     // cout<<(int)hIn[i * 2] * 0.007875937f<<" -- "<<(int)hOut[i] * 0.007875937f<<"\t";
        //     cout<<(int)hOut[i]<<" -- "<<(int)hOut[i] * 0.007875937f<<"\t";

        
        // delete[] hIn;
        // delete[] hOut;

    } else if(inDataType == DataType::kFLOAT) {
        const float* inData    = reinterpret_cast<const float*>(inputs[0]);
        float*       outBuffer = reinterpret_cast<float*>      (outputs[0]);

        size_t smemSize = sizeof(float) * imageWidth * readHeight;

        if(smemSize >= 49152) {
            cout<<"Error: Shared memory exceed the limit in Res2ChannelPadd"<<endl;
            return - 1;
        }

        Res2ChannelOptPaddingFloat<<<GridSize, BlockSize, sizeof(float) * imageWidth * readHeight>>>
                                    (inData, outBuffer, imageWidth, imageHeight, imageWidth / 2, imageHeight / 2,
                                    readHeight, channel, newWidth, newHeight, inOffset, outOffset);
    } else {
        cout<<"Current DataType is not supported in Res2ChannelPad"<<endl;

        return - 1;
    }

    // cout<<"Res2Channel Finish"<<endl;
    return 0;
}