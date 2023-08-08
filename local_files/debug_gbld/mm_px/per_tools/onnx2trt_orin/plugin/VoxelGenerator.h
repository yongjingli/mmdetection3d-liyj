

#ifndef __VOXEL_GENERATOR_HPP__
#define __VOXEL_GENERATOR_HPP__
#include "common.h"

#include <vector>

using namespace std;

#define VGBLOCKNUM(n, t) ((n - 1) / t + 1)

//Point 
struct PointXYZI {
    float x;
    float y;
    float z;
    float i;
};

struct VoxelParams {
    float _pcRangeUpper[3];
    float _pcRangeLower[3];
    float _resolution  [3];
    int   _inputShape  [3];

    void setParams() {
        _pcRangeUpper[0] = 51.2f;
        _pcRangeUpper[1] = 51.2f;
        _pcRangeUpper[2] = 3.6f;

        _pcRangeLower[0] = -51.2f;
        _pcRangeLower[1] = -51.2f;
        _pcRangeLower[2] = -2.0f;

        _resolution  [0] = 0.2f;
        _resolution  [1] = 0.2f;
        _resolution  [2] = 0.2f;

        _inputShape  [2] = (_pcRangeUpper[0] - _pcRangeLower[0]) / _resolution[0];
        _inputShape  [1] = (_pcRangeUpper[1] - _pcRangeLower[1]) / _resolution[1];
        _inputShape  [0] = (_pcRangeUpper[2] - _pcRangeLower[2]) / _resolution[2];

    }


    void setParams(vector<float> pcRangeUpper, vector<float> pcRangeLower, 
                   vector<float> resolution  , vector<int>   inputShape) {

        for(int i = 0; i < 3; ++i) {
            _pcRangeUpper[i] = pcRangeUpper[i];
            _pcRangeLower[i] = pcRangeLower[i];
            _resolution  [i] = resolution  [i];
            _inputShape  [i] = inputShape  [i];
        }
    }

    void setParams(vector<float> pcRangeUpper, vector<float> pcRangeLower, 
                   vector<float> resolution) { 

        for(int i = 0; i < 3; ++i) {
            _pcRangeUpper[i] = pcRangeUpper[i];
            _pcRangeLower[i] = pcRangeLower[i];
            _resolution  [i] = resolution  [i];
        }

        _inputShape  [2] = (_pcRangeUpper[0] - _pcRangeLower[0]) / _resolution[0];
        _inputShape  [1] = (_pcRangeUpper[1] - _pcRangeLower[1]) / _resolution[1];
        _inputShape  [0] = (_pcRangeUpper[2] - _pcRangeLower[2]) / _resolution[2];
    }

};


class VoxelGenerator {
public:
    VoxelGenerator(cudaStream_t stream, int dataType, float scale, bool isGpusData = false){
        // cout<<"VG is doing Consturctor."<<endl;
        _stream = stream;
        _vParams.setParams();
        _isGpuData = isGpusData;
        _scale = scale;
        _dataType = dataType;
        Initialization();
        // cout<<"VG constructor finish."<<endl;
    }

    VoxelGenerator(vector<float> pcRangeUpper, vector<float> pcRangeLower, 
                   vector<float> resolution, cudaStream_t stream, int dataType, 
                   float scale,  bool isGpusData = false) {
        // cout<<"VG is doing Consturctor."<<endl;
        _stream = stream;
        _isGpuData = isGpusData;
        _vParams.setParams(pcRangeUpper, pcRangeLower, resolution);
        _scale = scale;
        _dataType = dataType;
        Initialization();
        // cout<<"VG constructor finish."<<endl;
    }

    VoxelGenerator(vector<float> pcRangeUpper, vector<float> pcRangeLower, 
                   vector<float> resolution,   vector<int>   inputShape, 
                   cudaStream_t stream, int dataType, float scale, bool isGpusData = false) {
        _stream = stream;
        _isGpuData = isGpusData;
        _vParams.setParams(pcRangeUpper, pcRangeLower, resolution, inputShape);
        _scale = scale;
        _dataType = dataType;
        Initialization();
        
    }

    void setDataType(int dataType) {
        _dataType = dataType;
    }

    void setScale(float scale) {
        _scale = scale;
    }

    void copyData(PointXYZI* hPointData, int dataSize) {

        _pointSize = dataSize;
        if(_pointSize > _maxPointNum) {
            cout<<"The input data point number is greater than "<<_maxPointNum 
                <<". Truncate to "<<_maxPointNum<<" points"<<endl;
            _pointSize = _maxPointNum;
        }

        if(!_isGpuData) 
            cudaMemcpyAsync(_dPointData, hPointData, sizeof(PointXYZI) * _pointSize, cudaMemcpyHostToDevice, _stream);
        else 
            _dPointData = hPointData;

        // cout<<"The point size at copy data is "<<_pointSize<<endl;
        // cudaStreamSynchronize(_stream);
        // CHECK(cudaGetLastError());
    }

    void initWorkSpace() { 
        // cout<<"Init Work Space "<<endl;
        // cout<<"Is Gpu Data "<<_isGpuData<<endl;
        // cout<<"Data Type "<<_dataType<<endl;
        // cout<<"Scale is "<<_scale<<endl;
        // cudaMalloc(&_dVoxel,     sizeof(char)      * _pVoxelNum);
        // cudaMemsetAsync(_dVoxel, 0, sizeof(char) * _pVoxelNum, _stream);

        if(!_isGpuData)
            cudaMalloc(&_dPointData, sizeof(PointXYZI) * _maxPointNum);

        
        // cudaStreamSynchronize(_stream);
        // CHECK(cudaGetLastError());
    }

    void* generateVoxels(void* dVoxel);

    void* getValidVoxelCoors() {
        return _dVoxel;
    }

    int getTotVoxelNum() {
        return _pVoxelNum;
    }

    int getRealVoxelNum() {
        return _rVoxelNum;
    }

    void terminate() {

        if(_dPointData  != NULL && !_isGpuData) {
            cudaFree(_dPointData);
            _dPointData = NULL;
        }
            
        
    }

private:
    void Initialization() {
        // cout<<"Initializing VoxelGenerator"<<endl;

        if(_dataType == 2)
            _paddingHeight = ((_vParams._inputShape[0] - 1) / 32 + 1) * 32;
        else if(_dataType == 1)
            _paddingHeight = ((_vParams._inputShape[0] - 1) / 16 + 1) * 16;
        else
            _paddingHeight = _vParams._inputShape[0];

        _pVoxelNum = _paddingHeight          * _vParams._inputShape[1] * _vParams._inputShape[2];
        _rVoxelNum = _vParams._inputShape[0] * _vParams._inputShape[1] * _vParams._inputShape[2];

        // cout<<"Initializing Finish"<<endl;
    }

private:
    //Init Params
    vector<float> _pcRangeUpper;
    vector<float> _pcRangeLower;
    vector<float> _resolution;
    vector<float> _inputShape;
    
    VoxelParams _vParams;

    int   _maxPointNum = 100000;
    int   _pVoxelNum   = 0;
    int   _rVoxelNum   = 0;
    int   _pointSize   = 0;

    int   _dataType    = 2; //0: CHW GPU 1: DLA FP16 CHW16 2: DLA INT8 CHW32

    float _scale       = 1.0f;

    //Runtime params
    PointXYZI* _dPointData = NULL;

    void*  _dVoxel = NULL;

    int    _paddingHeight = 0;

    bool   _isGpuData = false;

    cudaStream_t _stream;
};


#endif