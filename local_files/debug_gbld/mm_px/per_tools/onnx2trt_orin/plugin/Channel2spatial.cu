#include <string.h>
#include "Channel2spatial.h"

#define SMEMSIZE 49152

// Static class fields initialization
PluginFieldCollection    Channel2SpatialCreator::mFC{};
std::vector<PluginField> Channel2SpatialCreator::mPluginAttributes;

struct TransposeParams {
    int outputExten[4];
    int inputExten [4];
    int smemExten  [2];

    int commonDivisor;
    int dimExpansion;

    int globalOffset;
    int heightWrtieOffset;
    int channelWriteOffset;


    void setParamsVector(vector<int> inputE, vector<int> outputE,
                         vector<int> smemE) {
        for(int i = 0; i < 4; ++ i) {
            inputExten [i] = inputE [i];
            outputExten[i] = outputE[i];
        }

        for(int i = 0; i < 2; ++ i) {
            smemExten  [i] = smemE  [i]; 
        }
    }

    void setParamsScalar(int commonD, int dimE) {
        commonDivisor = commonD;
        dimExpansion  = dimE;
    }

    void computeOffset() {
        globalOffset      = inputExten[3] * inputExten[2] * inputExten[1];
        heightWrtieOffset = inputExten[3] * commonDivisor;
        channelWriteOffset= inputExten[3] * inputExten[2] * commonDivisor;
    }
};

__forceinline__ __device__ int devSub2Ind(int* pos, TransposeParams& transParams) {

    return pos[3] + pos[2] * transParams.inputExten[3] + 
           pos[1] * transParams.inputExten[3]
                  * transParams.inputExten[2];
}

__forceinline__ __device__ int devSub2IndSmem(int* pos, TransposeParams& transParams) {
    
    return pos[1] + pos[0] * transParams.smemExten[1];
}


template<typename T>
__global__ void channel2spatial(T* oData, const T* iData, TransposeParams transParams, int readNum) {
    
    // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    extern __shared__ float smem[];
    T *dataBuff = reinterpret_cast<T *>(smem);

    int globalOffset = blockIdx.z * transParams.globalOffset; 

    int globalWrtie = blockIdx.y * transParams.heightWrtieOffset
                    + blockIdx.x * transParams.channelWriteOffset;

    globalWrtie += globalOffset;
                        
    for(int i = threadIdx.x; i < readNum; i += blockDim.x) {
        int inpos  [4];
        int smemPos[2];

        int channel = i / transParams.inputExten[3];
        int width   = i % transParams.inputExten[3];

        inpos[0] = 0; //N
        inpos[1] = channel + blockIdx.x * transParams.commonDivisor; //C
        inpos[2] = blockIdx.y; //H
        inpos[3] = i % transParams.inputExten[3]; //W

        int readPos = devSub2Ind(&inpos[0], transParams) + globalOffset;

        T val = iData[readPos];

        smemPos[0] = channel / transParams.dimExpansion; //H
        smemPos[1] = width   * transParams.dimExpansion 
                   + channel % transParams.dimExpansion; //W

        int smemIdx = devSub2IndSmem(&smemPos[0], transParams);

        dataBuff[smemIdx] = val;
        
    }

    __syncthreads();

    for(int i = threadIdx.x; i < readNum; i += blockDim.x) {
        
        oData[globalWrtie + i] = dataBuff[i];
    }
}

int getPower(int n, int cnt);

int getThraedNum(int n);
                            
int Channel2SpatialForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                           int scale, int mixed, std::vector<int> inputDims, DataType dType, 
                           TensorFormat dFormat, cudaStream_t stream) {
    cudaDebugError(2);

    if (mixed > 0) {// mixed batch for side
      return MixedBatchForward(batchSize, inputs, outputs, workspace, scale, mixed, inputDims, dType, dFormat, stream);
    }
    
    vector<int> dimExtensions  (4, 0);

    vector<int> dimExtensionsOut (4, 0);

    vector<int> dimSmemExtension(4, 0);

    int N = batchSize;
    int C = inputDims[0];
    int H = inputDims[1];
    int W = inputDims[2];

    int commonDivisor = scale;
    int dimExpansion  = sqrt(commonDivisor);
    int channelNew = C / commonDivisor;

    dimExtensions[0] = N;
    dimExtensions[1] = C;
    dimExtensions[2] = H;
    dimExtensions[3] = W;

    dimExtensionsOut[0] = N;
    dimExtensionsOut[1] = channelNew;
    dimExtensionsOut[2] = H * dimExpansion;
    dimExtensionsOut[3] = W * dimExpansion;

    dimSmemExtension[0] = dimExpansion;
    dimSmemExtension[1] = W * dimExpansion;

    //-----------Set C2S Params-----------//
    TransposeParams hCSTransposeParams;

    hCSTransposeParams.setParamsVector(dimExtensions, dimExtensionsOut, dimSmemExtension);
    hCSTransposeParams.setParamsScalar(commonDivisor, dimExpansion);
    hCSTransposeParams.computeOffset();

    //-----------Assign GPU 3D Space-----------//
    int readNum = W * commonDivisor;
    int power = getThraedNum(readNum);
    int threadX = pow(2, power - 1);

    threadX = threadX > 1024 ? 1024 : threadX;

    int blockNumX = channelNew;
    int blockNumY = H;
    int blockNumZ = N;

    dim3 c2sGrid (blockNumX, blockNumY, blockNumZ);
    dim3 c2sBlock(threadX,  1, 1);

    int sharedMem = SMEMSIZE + 1;

    if (TRT_DEBUGLEVEL == -1) {
            printf("The data type is %d\n", (int)dType);
            cout<<"Thread is "<<threadX<<endl;
            cout<<"Block num is "<<blockNumX<<"; "<<blockNumY<<"; "<<blockNumZ<<endl;

            for(int i = 0; i < 4; ++i) {
                cout<<"oldDimension ["<<i<<"] is "<<dimExtensions[i]<<"\t";
                cout<<"newDimension ["<<i<<"] is "<<dimExtensionsOut[i]<<"\n";
                // cout<<"smemDimension ["<<i<<"] is "<<dimExtensions[i]<<"\t";
            }
                
            // return 0;
    }



    if(dType == DataType::kINT8) {
        sharedMem = sizeof(char) * readNum;
        if(sharedMem > SMEMSIZE) {
            cout<<"Error: The shared memory exceed the limit. Return."<<endl;
            assert(sharedMem <= SMEMSIZE);
        }
        const char* inDataDev = reinterpret_cast<const char*>(inputs[0]);
        char*       outBuff   = reinterpret_cast<char*>      (outputs[0]);

        channel2spatial<char><<<c2sGrid, c2sBlock, sharedMem, stream>>>(outBuff, inDataDev, hCSTransposeParams, readNum);

    }else if(dType == DataType::kHALF) {
        sharedMem = sizeof(uint16_t) * readNum;
        if(sharedMem > SMEMSIZE) {
            cout<<"Error: The shared memory exceed the limit. Return."<<endl;
            assert(sharedMem <= SMEMSIZE);
        }

        const uint16_t* inDataDev = reinterpret_cast<const uint16_t*>(inputs[0]);
        uint16_t*       outBuff   = reinterpret_cast<uint16_t*>      (outputs[0]);
        channel2spatial<uint16_t><<<c2sGrid, c2sBlock, sharedMem, stream>>>(outBuff, inDataDev, hCSTransposeParams, readNum);
    }else {
        sharedMem = sizeof(float) * readNum;

        if(sharedMem > SMEMSIZE) {
            cout<<"Error: The shared memory exceed the limit. Return."<<endl;
            assert(sharedMem <= SMEMSIZE);
        }

        const float* inDataDev = reinterpret_cast<const float*>(inputs[0]);
        float*       outBuff   = reinterpret_cast<float*>      (outputs[0]);
    
        channel2spatial<float><<<c2sGrid, c2sBlock, sharedMem, stream>>>(outBuff, inDataDev, hCSTransposeParams, readNum);
        // channel2spatial<<<c2sGrid, c2sBlock, sharedMem, stream>>>(outBuff, inDataDev, hCSTransposeParams, readNum);
    }

    cudaDebugError(2);
    return cudaGetLastError(); 
}

int getPower(int n, int cnt) {
    n = n >> 1;
    return cnt = n == 0 ? cnt + 1: getPower(n, cnt + 1);
}

int getThraedNum(int n) {
    int cnt = 0;
    return getPower(n, cnt);
}


// mixed batch for side: side_front_left x side_rear_left, side_front_right x side_rear_right, 
template<int SCALE, typename T>
__global__ void MixedBatchKernel(int batchSize, const T* idata, int4 inDims, T* odata, int4 outDims) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int ob = 0; ob < batchSize; ob ++){
    for (int oc = z0; oc < outDims.z; oc += blockDim.z * gridDim.z ) {
      for (int oy = y0; oy < outDims.y; oy += blockDim.y * gridDim.y ) {
        for (int ox = x0; ox < outDims.x; ox += blockDim.x * gridDim.x ) {
          int ib = ob / 2 * 2 + oc / inDims.x;
          int ic = ob % 2 * outDims.x + ox;
          int ix = oc % inDims.x;

          int idx_i = ((ib * inDims.z + ic) * inDims.y + oy) * inDims.x + ix;
          int idx_o = ((ob * outDims.z + oc) * outDims.y + oy) * outDims.x + ox;
          odata[idx_o] = idata[idx_i];
          //if( idx_o % 1000000 < 10 )  printf("%d <- %d\n", idx_o, idx_i);
        }
      }
    }
  }
}

template<int SCALE, typename T>
__global__ void MixedBatchKernel_CHW32(int batchSize, const T* idata, int4 inDims, T* odata, int4 outDims) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int ob = 0; ob < batchSize; ob ++){
    for (int oc = z0; oc < outDims.z; oc += blockDim.z * gridDim.z ) {
      for (int oy = y0; oy < outDims.y; oy += blockDim.y * gridDim.y ) {
        for (int ox = x0; ox < outDims.x; ox += blockDim.x * gridDim.x ) {
          int ib = ((ob >> 1) << 1) + oc / inDims.x;
          int ic = (ob & 1) * outDims.x + ox;
          int ix = oc % inDims.x;
          int idx_i = (((((((ib * inDims.z) >> 5) + (ic >> 5)) ) * inDims.y + oy) * inDims.x) << 5) + (ix << 5) + (ic & 31);
          int idx_o = (((((((ob * outDims.z) >> 5) + (oc >> 5)) ) * outDims.y + oy) * outDims.x) << 5) + (ox << 5) + (oc & 31);
          odata[idx_o] = idata[idx_i];
        }
      }
    }
  }
}

int MixedBatchForward(int batchSize, const void* const *inputs, void* const* outputs, void* workspace,
                      int scale, int mixed, std::vector<int> inputDims, DataType dType, 
                      TensorFormat dFormat, cudaStream_t stream){

    int4 inDims = {inputDims[2], inputDims[1], inputDims[0], batchSize}; // x, y, z, w
    int4 outDims = {inputDims[0]/scale, inputDims[1], inputDims[2]*scale, batchSize}; // x, y, z, w

    dim3 block(4, 4, 16); // block.z = 16, inDims.x = 64/128/256, outDims.z = 32/64/128
    dim3 grid((outDims.x - 1) / block.x + 1, (outDims.y - 1) / block.y + 1, outDims.z / block.z);
    
    if(dType == DataType::kINT8) {
      MixedBatchKernel_CHW32<2, char><<<grid, block, 0, stream>>>(batchSize, (const char*)inputs[0], inDims, (char*)outputs[0], outDims);
    } else if(dType == DataType::kHALF) {
      MixedBatchKernel<2, half><<<grid, block, 0, stream>>>(batchSize, (const half*)inputs[0], inDims, (half*)outputs[0], outDims);
    } else if(dType == DataType::kFLOAT) {
      MixedBatchKernel<2, float><<<grid, block, 0, stream>>>(batchSize, (const float*)inputs[0], inDims, (float*)outputs[0], outDims);
    } else {
      DPRINTF(1, "MixedBatchForward Not support Type:%d Format:%d\n", (int)dType, (int)dFormat);
    }                   
    
    cudaDebugError(2);
    return cudaGetLastError();
}


// REGISTER_TENSORRT_PLUGIN(Channel2SpatialCreator);