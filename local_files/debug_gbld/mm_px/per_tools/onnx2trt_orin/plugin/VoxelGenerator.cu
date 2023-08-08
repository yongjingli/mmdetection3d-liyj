
#include "VoxelGenerator.h"

template<int CHNUM>
__global__ void cudaGetValidVoxelInt8(const PointXYZI* devPointData, signed char* dVoxel, 
                                        VoxelParams gpuVoxelParams, int voxelNum, int pointNum) {
    int idx     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride  = blockDim.x  * gridDim.x;

    for(int i = idx; i < pointNum; i +=stride ) {
        PointXYZI point = devPointData[i];

        if( point.x > gpuVoxelParams._pcRangeUpper[0] || point.x < gpuVoxelParams._pcRangeLower[0] ||
            point.y > gpuVoxelParams._pcRangeUpper[1] || point.y < gpuVoxelParams._pcRangeLower[1] ||
            point.z > gpuVoxelParams._pcRangeUpper[2] || point.z < gpuVoxelParams._pcRangeLower[2]) {
                continue;
        } else {
            int4 coor;
            coor.x = (int)floor(((point.x - gpuVoxelParams._pcRangeLower[0]) / gpuVoxelParams._resolution[0]));
            coor.y = (int)floor(((point.y - gpuVoxelParams._pcRangeLower[1]) / gpuVoxelParams._resolution[1]));
            coor.z = (int)floor(((point.z - gpuVoxelParams._pcRangeLower[2]) / gpuVoxelParams._resolution[2]));

            // int pos = coor.z * gpuVoxelParams._inputShape[1] * gpuVoxelParams._inputShape[2] +
            //           coor.y * gpuVoxelParams._inputShape[2] + coor.x;

            // pos = pos < 0 ? 0 : pos;
            // pos = pos >= voxelNum ? (voxelNum - 1): pos;

            // dVoxel[pos] = 127;

            int pos1 = coor.z * gpuVoxelParams._inputShape[1] * gpuVoxelParams._inputShape[2] +
                       coor.y * gpuVoxelParams._inputShape[2] + coor.x;

            coor.w = coor.z / CHNUM;

            int pos2 = coor.w * CHNUM * gpuVoxelParams._inputShape[2] * gpuVoxelParams._inputShape[1] +
                       coor.y * CHNUM * gpuVoxelParams._inputShape[2] + coor.x * CHNUM + coor.z % CHNUM;

            pos2 = pos1 < 0 ? 0 : pos2;
            pos2 = pos1 >= voxelNum ? (voxelNum - 1): pos2;
            
            dVoxel[pos2] = 127;
        }
    }
}


template<int CHNUM>
__global__ void cudaGetValidVoxelHalf(const PointXYZI* devPointData, __half* dVoxel, float scale,
                                        VoxelParams gpuVoxelParams, int voxelNum, int pointNum) {
    int idx     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride  = blockDim.x  * gridDim.x;

    for(int i = idx; i < pointNum; i +=stride ) {
        PointXYZI point = devPointData[i];

        if( point.x > gpuVoxelParams._pcRangeUpper[0] || point.x < gpuVoxelParams._pcRangeLower[0] ||
            point.y > gpuVoxelParams._pcRangeUpper[1] || point.y < gpuVoxelParams._pcRangeLower[1] ||
            point.z > gpuVoxelParams._pcRangeUpper[2] || point.z < gpuVoxelParams._pcRangeLower[2]) {
                continue;
        } else {
            int4 coor;
            coor.x = (int)floor(((point.x - gpuVoxelParams._pcRangeLower[0]) / gpuVoxelParams._resolution[0]));
            coor.y = (int)floor(((point.y - gpuVoxelParams._pcRangeLower[1]) / gpuVoxelParams._resolution[1]));
            coor.z = (int)floor(((point.z - gpuVoxelParams._pcRangeLower[2]) / gpuVoxelParams._resolution[2]));

            // int pos = coor.z * gpuVoxelParams._inputShape[1] * gpuVoxelParams._inputShape[2] +
            //           coor.y * gpuVoxelParams._inputShape[2] + coor.x;
 
            // pos = pos < 0 ? 0 : pos;
            // pos = pos >= voxelNum ? (voxelNum - 1): pos;
            
            // dVoxel[pos] = __float2half(scale);

            int pos1 = coor.z * gpuVoxelParams._inputShape[1] * gpuVoxelParams._inputShape[2] +
                       coor.y * gpuVoxelParams._inputShape[2] + coor.x;

            coor.w = coor.z / CHNUM;

            int pos2 = coor.w * CHNUM * gpuVoxelParams._inputShape[2] * gpuVoxelParams._inputShape[1] +
                       coor.y * CHNUM * gpuVoxelParams._inputShape[2] + coor.x * CHNUM + coor.z % CHNUM;

            pos2 = pos1 < 0 ? 0 : pos2;
            pos2 = pos1 >= voxelNum ? (voxelNum - 1): pos2;
            
            dVoxel[pos2] = __float2half(scale);
        }
    }
}

template<int CHNUM>
__global__ void cudaGetValidVoxelFloat(const PointXYZI* devPointData, float* dVoxel, float scale,
                                        VoxelParams gpuVoxelParams, int voxelNum, int pointNum) {
    int idx     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride  = blockDim.x  * gridDim.x;

    for(int i = idx; i < pointNum; i +=stride ) {
        PointXYZI point = devPointData[i];

        if( point.x > gpuVoxelParams._pcRangeUpper[0] || point.x < gpuVoxelParams._pcRangeLower[0] ||
            point.y > gpuVoxelParams._pcRangeUpper[1] || point.y < gpuVoxelParams._pcRangeLower[1] ||
            point.z > gpuVoxelParams._pcRangeUpper[2] || point.z < gpuVoxelParams._pcRangeLower[2]) {
                continue;
        } else {

            int4 coor;
            coor.x = (int)floor(((point.x - gpuVoxelParams._pcRangeLower[0]) / gpuVoxelParams._resolution[0]));
            coor.y = (int)floor(((point.y - gpuVoxelParams._pcRangeLower[1]) / gpuVoxelParams._resolution[1]));
            coor.z = (int)floor(((point.z - gpuVoxelParams._pcRangeLower[2]) / gpuVoxelParams._resolution[2]));

            // coor.x = coor.x >= gpuVoxelParams._inputShape[2] ? gpuVoxelParams._inputShape[2] - 1 : coor.x;
            // coor.y = coor.y >= gpuVoxelParams._inputShape[1] ? gpuVoxelParams._inputShape[1] - 1 : coor.y;
            // coor.z = coor.z >= gpuVoxelParams._inputShape[0] ? gpuVoxelParams._inputShape[0] - 1 : coor.z;
            // coor.y = coor.y < 0 ? 0 : coor.y;
            // coor.x = coor.x < 0 ? 0 : coor.x;
            // coor.z = coor.z < 0 ? 0 : coor.z;

            int pos = coor.z * gpuVoxelParams._inputShape[1] * gpuVoxelParams._inputShape[2] +
                      coor.y * gpuVoxelParams._inputShape[2] + coor.x;

            pos = pos < 0 ? 0 : pos;
            pos = pos >= voxelNum ? (voxelNum - 1): pos;
            
            dVoxel[pos] = scale; 

            // printf("%d -- %d;\t", i, pos);
        }
    }
}

void* VoxelGenerator::generateVoxels(void* dVoxel) {
    // cout<<"Generating Voxel"<<endl;
    // cout<<"The point num is "<<_pointSize<<endl;
    _dVoxel = dVoxel;
    
    if       (_dataType == 2) {
        cudaMemsetAsync(_dVoxel, 0, sizeof(char) * _rVoxelNum, _stream);
        cudaGetValidVoxelInt8<32><<<VGBLOCKNUM(_pointSize, 1024), 1024, 0, _stream>>>
            (_dPointData, reinterpret_cast<signed char*>(_dVoxel), _vParams, _rVoxelNum, _pointSize);
    } else if(_dataType == 1) { 
        cudaMemsetAsync(_dVoxel, 0, sizeof(__half) * _rVoxelNum, _stream);
        cudaGetValidVoxelHalf<16><<<VGBLOCKNUM(_pointSize, 1024), 1024, 0, _stream>>>
            (_dPointData, reinterpret_cast<__half*>(_dVoxel), 1.0f / _scale, _vParams, _rVoxelNum, _pointSize);
    } else if(_dataType == 0) { 
        cudaMemsetAsync(_dVoxel, 0, sizeof(float) * _rVoxelNum, _stream);
        cudaGetValidVoxelFloat<1><<<VGBLOCKNUM(_pointSize, 1024), 1024, 0, _stream>>>
        (_dPointData, reinterpret_cast<float*>(_dVoxel),  1.0f / _scale, _vParams, _rVoxelNum, _pointSize);
    } else {
        cout<<"Error: the data type is not supported. Return nullptr."<<endl;
        _dVoxel = nullptr;
    }

    // CHECK(cudaGetLastError());

    return _dVoxel;
}  