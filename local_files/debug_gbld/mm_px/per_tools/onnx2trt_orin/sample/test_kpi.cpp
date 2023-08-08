#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>  // For ::getopt
#include "cnpy.h"
#include "onnxtrt.h"
#include "ResizeBilinear.hpp"
#include <string>
#include <sstream>
//system operation
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define DICMODE (S_IRWXU | S_IRWXG | S_IRWXO) 

using namespace std;

PTCreateEngine pCreateEngine;
PTRunEngine pRunEngine;
PTDestoryEngine pDestoryEngine;
PTGetBufferOfEngine pGetBuffer;

PTAllocateSpaceGPU pAllocateSpaceGPU;
PTMemcpyHost2DeviceGPU pMemcpyHost2DeviceGPU;
PTFreeSpaceGPU pFreeSpaceGPU;

static int batchSize  = 1;
static int outputSize = 0;


void printUsage() {
  cout << "Test Kpi" << endl;
  cout << "\n\t[-s libonnxtrt.so (path of libonnxtrt.so) ] "
       << "\n\t[-e engine_file.trt  (test TensorRT single engine) ]"
       << "\n\t[-i input_data.txt (input png list stored in UNIX format) ]"
       << "\n\t[-o output path (the output path to saved spcified output type) ]"
       << "\n\t[-m modelConfig ( ex: Task=LLD|MOD|KPTL,Prioity=High,CudaGraph=True) ]"
       << "\n\t[-t model type ( ex: -t lidar/image) ]"              
       << "\n\t[-p is input data already preprocessed. ]"   
       << "\n\t[-w output file type. ( ex: -w bin/npy) ]"
       << "\n\t[-O The output type need to be saved.] " 
       << "\n\t[-G input data type. ( ex: -G 4 is for lidar; -G 0 is for normal cpu input) ]"
       << "\n\t[-S the scale for voxel generator (ex. 255.) ]"
       << "\n\t[-g is voxel generator need to be done in gpu ]"
       << "\n\t[-v is the data of generated voxel need to be saved ]"
       << "\n\t build at "<<__DATE__<< endl;
}



int getPreprocesParams(int &modelType, int &imageWidth, int &imageHeight, int *idxRGB,
                       float *Scales, float *Means) {
    // check for Width & Height
    char *val = getenv("TRT_IW");
    if (NULL != val) {
        imageWidth = atoi(val);
        printf("getenv TRT_IW=%d\n", imageWidth);
    }

    val = getenv("TRT_IH");
    if (NULL != val) {
        imageHeight = atoi(val);
        printf("getenv TRT_IH=%d\n", imageHeight);
    }

    val = getenv("TRT_BGR");
    if (NULL != val) {
        int isBGR = atoi(val);
        printf("getenv TRT_BGR=%d\n", isBGR);
        if (isBGR) {
        idxRGB[0] = 2;
        idxRGB[2] = 0;
        }
    }

    if (1 == modelType) { // FSD
        Means[0] = 102.9801f;  // R
        Means[1] = 115.9465f;  // G
        Means[2] = 122.7717f;  // B
    }
    
    if (2 <= modelType) { // LLDMOD / AP+FSD
        Scales[0] = 255.f;  // R
        Scales[1] = 255.f;  // G
        Scales[2] = 255.f;  // B
    }

    return 0;
}

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int readImageTo12Channel(std::vector<std::string> fileNameVector, unsigned char *&rgbBuffer, int &imageWidth,int &imageHeight, 
                         int &imageChannel, char *&dataBuffer, float &imageScale, int inputType, int modelType = 1) {
    
    float *floatBuffer = nullptr;
    for(int nB = 0; nB < batchSize; ++nB) {
        const char* fileName = fileNameVector[nB].c_str();
        int channel = 3;
        rgbBuffer = stbi_load(fileName, &imageWidth, &imageHeight, &channel, 0);
        int realChannel  = channel;
        channel = channel == 4 ? 3 : channel;
        imageChannel = channel;

        int imageSize = imageWidth * imageHeight;
        int size      = imageSize * channel;
        int realSize = imageSize * realChannel ;

        printf("readImage %s by stbi_load, size = %d (%dx%dx%d)\n", fileName, realSize, realChannel , imageHeight, imageWidth);

        if( imageWidth == 0 || imageHeight == 0 ){
            fprintf(stderr, "readImageByOpenCV read image fail: %s\n", fileName);
            return -1;
        }  

        int posOffset = nB * size;

        if (nB == 0) {
             //Allocate Buffer in Memory
            if(inputType == 3 || inputType == 5) 
                dataBuffer = new char[size * batchSize];
            else
                floatBuffer = new float[size * batchSize];
        } 

        int idxRGB[3] = {0, 1, 2};

        if(inputType == 3 || inputType == 5) {

            for (int i = 0; i < imageSize; ++i) 
                for (int j = 0; j < channel; ++j)
                    dataBuffer[(i + j * imageSize) + posOffset] =(reinterpret_cast<char*>(rgbBuffer)[i * realChannel  + idxRGB[j]]);

            continue;
        }

        float Scales[3] = {1.f, 1.f, 1.f};
        float Means[3]  = {0.f, 0.f, 0.f};

        int modelWidth  = 0;
        int modelHeight = 0;

        getPreprocesParams(modelType, modelWidth, modelHeight, idxRGB, Scales, Means);
        int qtImageSize = modelWidth * modelHeight;

        float* floatBuffBatch = floatBuffer + posOffset ;

        float *tmpBuffer = new float[size];

        for (int i = 0; i < imageSize; i++) {
            for(int j = 0; j < imageChannel; ++j) {
                tmpBuffer[i + imageSize * j] = (rgbBuffer[i * realChannel + idxRGB[j]] - Means[j]) / Scales[j]; 
            }
        }

        // [h, w, 3] -> [12, h/2, w/2]
        int srcIdx = 0;
        for( int c = 0; c < imageChannel; c++){
            for( int y = 0; y < imageHeight; y++) {
                for( int x = 0; x< imageWidth; x++) {
                    int dstC = ((x & 1) * 2 + (y & 1)) * imageChannel + c;
                    int dstY = y / 2;
                    int dstX = x / 2;
                    int dstIdx = dstC * (imageWidth / 2) * (imageHeight / 2) + dstY * (imageWidth / 2) + dstX;    

                    floatBuffBatch[dstIdx] = tmpBuffer[srcIdx ++];
                }
            }
        }

        delete[] tmpBuffer;

        if(rgbBuffer != NULL) {
            delete[] rgbBuffer;
            rgbBuffer = nullptr;
        } 
    }

    if(inputType != 3 && inputType != 5)
        dataBuffer = (char *)floatBuffer;

    return 0;
}

int readBinFile(const char *fileName, char *&dataBuffer) {
   int size = 0;
   std::ifstream file(fileName, std::ifstream::binary);
   if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        dataBuffer = new char[size];
        file.read(dataBuffer, size);
        file.close();
    }
    return size;
}

// struct PointXYZI {
//     double x;
//     double y;
//     double z;
//     double i;
// };

struct PointXYZI {
    float x;
    float y;
    float z;
    float i;
};


int voxelGenerator(std::string fileName, char *&dataBuffer, std::string dicName, float scale, bool saveVoxel) {

    cout<<"Generating Voxel..."<<endl;

    // vector<float> pcRangeUpper = { 51.2f,  51.2f,  3.61f};
    // vector<float> pcRangeLower = {-51.2f, -51.2f, -2.0f};
    // vector<float> resolution   = { 0.2f,   0.2f,   0.2f};
    // vector<float> inputShape   = { 28,     512,    512};

    // vector<float> pcRangeLower = {-0.5f,  -24.0f, -1.0f};
    // vector<float> pcRangeUpper = { 47.5f,  24.0f,  3.64f};
    // vector<float> resolution   = { 0.25f,   0.25f,   0.145f};
    // vector<float> inputShape   = { 32,     192,    192};

    vector<float> pcRangeLower = {-0.5f,  -36.0f, -1.0f};
    vector<float> pcRangeUpper = { 71.5f,  36.0f,  3.64f};
    vector<float> resolution   = { 0.25f,   0.25f,   0.145f};
    vector<float> inputShape   = { 32,     288,    288};

    // vector<double> pcRangeUpper = { 47.5d,  24.0d,  3.64d};
    // vector<double> pcRangeLower = {-0.5d,  -24.0d, -1.0d};
    // vector<double> resolution   = { 0.25d,   0.25d,   0.145d};
    // vector<double> inputShape   = { 32,     192,    192};

    int   size       = 0;
    char* pcBuffChar = NULL;

    //Read original poing cloud from bin file.
    std::ifstream file(fileName.c_str(), std::ifstream::binary);
    if(file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        pcBuffChar = new char[size];
        file.read(pcBuffChar, size);
        file.close();
    }
    if(size <= 0) {
        cout<<"ERROR at voxelGenerator: Read "<<fileName<<" failed. Return"<<endl;
        return -1;
    }

    int pN = size / sizeof(PointXYZI);
    int vN = inputShape[0] * inputShape[1] * inputShape[2];

    cout<<"The Point Num is "<<pN<<endl;
    cout<<"The Voxel Num is "<<vN<<endl;

    dataBuffer = new char[vN * 4];

    float*     voxelBuff = reinterpret_cast<float*>    (dataBuffer);
    PointXYZI* pcBuff    = reinterpret_cast<PointXYZI*>(pcBuffChar);

    for(int i = 0; i < vN; ++i) 
        voxelBuff[i] = 0.0f;


    for(int i = 0; i < pN; ++i) {
        PointXYZI point = pcBuff[i];
        if(point.x > pcRangeUpper[0] || point.x < pcRangeLower[0] ||
           point.y > pcRangeUpper[1] || point.y < pcRangeLower[1] ||
           point.z > pcRangeUpper[2] || point.z < pcRangeLower[2]) {
               continue;
        } else {
           int x = (int)floor(((point.x - pcRangeLower[0]) / resolution[0]));
           int y = (int)floor(((point.y - pcRangeLower[1]) / resolution[1]));
           int z = (int)floor(((point.z - pcRangeLower[2]) / resolution[2]));

           int pos = 0; 
           pos = z * inputShape[1] * inputShape[2] + y * inputShape[2] + x;

           if(pos < 0 )  pos = 0;
           if(pos >= vN) pos = vN - 1;

           voxelBuff[pos] = 1.0f;
        }
    }

    for(int i = 0; i < vN; ++i) 
        voxelBuff[i] = voxelBuff[i] / scale;

    if(saveVoxel) {
        cout<<"Saving generated voxel into bin file."<<endl;
        std::string::size_type iPos = fileName.find_last_of('/') + 1;
        std::string fName = fileName.substr(iPos, fileName.length() - iPos);
        std::string name = fName.substr(0, fName.rfind("."));

        std::string outFilePath = dicName + "/" + name + ".bin";
        cout<<"The bin file name of generated voxel is: "<<outFilePath<<endl;
        std::ofstream out(outFilePath.c_str(), std::ios_base::binary);
        if(out.good()) {
            out.write((const char*)dataBuffer, vN * 4);
            out.close();
        }else {
            cout<<"ERROR: Cannot write to binary file. Return."<<endl;
        }

        cout<<"Wrote generated voxel to bin done."<<endl;
    }

    if(pcBuffChar != NULL) delete[] pcBuffChar;

    return 1;

}

// process the output of default model-----------------------------------------
void writResultNumpy(float *opData, std::string outFile, std::string inputFile, std::string outClass,
                       int imageWidth, int imageHeight,  int opOffset, int writeNum) {
    // const int imageSize = imageWidth * imageHeight;

    string::size_type iPos = inputFile.find_last_of('/') + 1;
	string fileName = inputFile.substr(iPos, inputFile.length() - iPos);
 
	string name = fileName.substr(0, fileName.rfind("."));

    // float *writeData = opData + opOffset;

    unsigned int valNum = static_cast<unsigned int>(writeNum);

    cout<<"writing class "<<outClass<<endl;
    cout<<"offset pos is "<<opOffset<<endl;
    cout<<"wrting num is "<<valNum<<endl;

    std::string outFileName = outFile + "/" + name + "-" + outClass + ".npy";
    
    cnpy::npy_save(outFileName, &opData[opOffset], {valNum});

    cout<<"Wrote "<<inputFile<<" result to "<<outFileName<<" done."<<endl;
    
}

// process the output of default model-----------------------------------------
void writeResultBin(float *opData, std::string outFile, std::string inputFile, std::string outClass,
                    int imageWidth, int imageHeight,  int opOffset, int writeNum) {
    // const int imageSize = imageWidth * imageHeight;

    string::size_type iPos = inputFile.find_last_of('/') + 1;
	string fileName = inputFile.substr(iPos, inputFile.length() - iPos);
 
	string name = fileName.substr(0, fileName.rfind("."));

    float *writeData = opData + opOffset;

    unsigned int valNum = static_cast<unsigned int>(writeNum);
    cout<<"writing class "<<outClass<<endl;
    cout<<"offset pos is "<<opOffset<<endl;
    cout<<"wrting num is "<<valNum<<endl;
    std::string outFileName = outFile + "/" + name + "-" + outClass + ".bin";
    std::ofstream out(outFileName, std::ios_base::binary);
    if(out.good()) {
        out.write((const char*)writeData, valNum *  sizeof(float));
        out.close();
    }else {
        cout<<"ERROR: Cannot write to binary file. Return."<<endl;
    }
    cout<<"Wrote "<<inputFile<<" result to "<<outFileName<<" done."<<endl;
}

bool checkValidImageFormatImage(std::string& inputFileName, int& nameLength) {
    if( inputFileName.compare(nameLength, 4, ".ppm") == 0 ||
        inputFileName.compare(nameLength, 4, ".png") == 0 ||
        inputFileName.compare(nameLength, 4, ".jpg") == 0 ||
        inputFileName.compare(nameLength, 4, ".png") == 0 ||
        inputFileName.compare(nameLength, 4, ".bpm") == 0)
        return true;
    else
        return false;
}

bool checkValidImageFormatBin(std::string& inputFileName, int& nameLength) {
    if( inputFileName.compare(nameLength, 4, ".bin") == 0)
        return true;
    else
        return false;
}

bool checkTxtFormat(std::string& inputFileName, int nameLength) {
    if( inputFileName.compare(nameLength, 4, ".txt") == 0)
        return true;
    else
        return false;
}

void buildFolder(std::string dicName) {
    //check if the outpath exist
    struct stat opSt;

    if(stat(dicName.c_str(), &opSt) != 0) {
        int ret = mkdir(dicName.c_str(), DICMODE);
        if(ret < 0) {
            cout<<"ERROR: Folder created failed. Return."<<endl;
        } else {
            cout<<"Folder created successed!"<<endl;
            cout<<"The output of the model will be saved at "<<dicName<<endl;
        }
    } else {
        cout<<"The output of the model will be saved at "<<dicName<<endl;
    }
}


int main(int argc, char *argv[])
{
    if (argc <= 1) printUsage();
    const char *pSoName       = "./libonnxtrt.so";
    std::string outPath       = "./";
    std::string writeType     = "npy";
    std::string engineFile;
    std::string maskWeight;

    std::vector<std::string> outClass;
    std::vector<std::string> inputFile;
    
    LidarInput lidarInput;
    int inputType = 0;
    float scale = 1.0f;

    bool isLidarModel  = false;
    bool dataPPed      = false;
    bool isNpy         = false;
    bool saveVoxel     = false;
    bool needGpuProcess= false;
    bool loopMode      = false;
    bool sequenceMode  = false;

    //get command args 
    int arg = 0;
    const char *optString = "s:e:i:o:m:O:w:G:S:b:pgvltT";

    while ((arg = ::getopt(argc, argv, optString)) != -1) {
        if ('e' == arg || 'i' == arg || 's' == arg || 'o' == arg || 'm' == arg || 
            'O' == arg || 't' == arg || 'p' == arg || 'w' == arg || 'G' == arg || 
            'S' == arg || 'b' == arg) {
            if (!optarg) {
                cerr << "ERROR: -" << arg << " flag requires argument" << endl;
                return -1;
            }
        }
        switch (arg) {
        case 'e':
            engineFile     = optarg;
            break;        
        case 'i':
            inputFile.push_back(optarg);
            break;
        case 's':
            pSoName        = optarg;
            break;
        case 'b':
            batchSize      = atoi(optarg);
            break;
        case 'o':
            outPath        = optarg;
            break;
        case 'm':
            maskWeight     = optarg;
            break;
        case 't':
            isLidarModel   = true;
            break;
        case 'p':
            dataPPed       = true;
            break;
        case 'w':
            writeType      = optarg;
            break;
        case 'v':
            saveVoxel      = true;
            break;
        case 'S':
            scale          = atof(optarg);
            break;
        case 'O':
            outClass.push_back(optarg);
            break;
        case 'G':
            inputType      = atoi(optarg);
            break;
        case 'g':
            needGpuProcess = true;
            break;
        case 'l':
            loopMode       = true;
            break;
        case 'T':
         sequenceMode      = true;
            break;
        }
    }

    void *pLibs = dlopen(pSoName, RTLD_LAZY);
    if (pLibs == NULL) {
        printf("Can not open library %s\n", pSoName);
        return -1;
    }

    if((needGpuProcess && inputType !=4) || (inputType ==4 && !needGpuProcess)) {
        cout<<"The input type and gpu voxel process flags mismatch! "<<endl;
        return -1;
    }

    isNpy         = (NULL != strcasestr("npy",   writeType.    c_str()))? true : false;

    pCreateEngine  = (PTCreateEngine)     dlsym(pLibs, "CreateEngine");
    pRunEngine     = (PTRunEngine)        dlsym(pLibs, "RunEngine");
    pDestoryEngine = (PTDestoryEngine)    dlsym(pLibs, "DestoryEngine");
    pGetBuffer     = (PTGetBufferOfEngine)dlsym(pLibs, "GetBufferOfEngine");

    pAllocateSpaceGPU     = (PTAllocateSpaceGPU)dlsym(pLibs, "AllocateSpaceGPU");
    pMemcpyHost2DeviceGPU = (PTMemcpyHost2DeviceGPU)dlsym(pLibs, "MemcpyHost2DeviceGPU");
    pFreeSpaceGPU         = (PTFreeSpaceGPU)dlsym(pLibs, "FreeSpaceGPU");

    if(pFreeSpaceGPU == NULL && pMemcpyHost2DeviceGPU == NULL && pAllocateSpaceGPU == NULL) 
        cout<<"Warning: GPU memory functions cannot be load. The input type 2 and 3 are disabled."<<endl;

    //check file path format
    if(outPath[outPath.size() - 1] == '/') 
        outPath.erase(outPath.size() - 1);

    //build general output path
    buildFolder(outPath);

    //Creat folder to save generated voxel
    std::string dicName;

    if(saveVoxel) {
        struct stat st;
        dicName = outPath + "/voxel";
        if(stat(dicName.c_str(), &st) != 0) {
            int ret = mkdir(dicName.c_str(), DICMODE);
            if(ret < 0) {
                cout<<"ERROR: Folder created failed. Return."<<endl;
                return -1;
            } else {
                cout<<"Folder created successed!"<<endl;
                cout<<"The generated voxel will be saved at "<<dicName<<endl;
            }
        } else {
            cout<<"Warning: The folder has already existed, won't create another one. The previous files will be overwriten"<<endl;
        }
        // if(st.st_mode & S_IFDIR == 0) {
        //     cout<<"Warning: The folder has already existed, won't create another one. The previous files will be overwriten"<<endl;
        // }
    }


    if(inputFile.size() != batchSize) {
        cout<<"ERROR: The number of input lists for each batch are not equal. Retrun."<<endl;
        return -1;
    }
    
    //Read Txt List
    std::vector<std::string> mTxtList;
    std::vector<std::string>::iterator it;
    
    int initLineCounter = -1; //This variable describe how many txt files input for one batch.
    //Read all txt lists and check the corresponding txt lists number for each batch are equal.
    for(it = inputFile.begin(); it != inputFile.end(); it++) {
        int curLineCounter = 0;
        std::ifstream inList(*it);
        if(loopMode) {
            if (inList.good()) {
                std::string imgListPath;
                while (std::getline(inList, imgListPath)) {
                    if (imgListPath.length() > 2) mTxtList.push_back(imgListPath);
                    curLineCounter ++;
                }
                if(initLineCounter == -1) {
                    initLineCounter = curLineCounter;
                } else {
                    if (curLineCounter != initLineCounter) {
                        cout<<"ERROR: The number of input lists for each batch are not equal. Retrun."<<endl;
                        return -1;
                    }
                }
            } else{
                cout<<"ERROR: Open "<<*it<<" failed"<<endl;
            }
        }else {   
            initLineCounter = 1;
            mTxtList.push_back(*it);
        }
    }

    //declare all the vars need to be used as engine inference
    int engineID  = -1;
    int outType   = 0; // 0: plan 2 : packed by batchsize
    int modelType = 9; //Default mode 9 for KPI test
    const char *trtModel = engineFile.c_str();
    const char *pMaskWeight = NULL;

    cout<<"/------------Currently Testing "<<engineFile.c_str()<<"------------/"<<endl;

    if (pCreateEngine != NULL && pRunEngine != NULL && pDestoryEngine != NULL) {
        
        int ret = 0;
        //1.Create Engine
        cout<<"/------------Creating Engine Located in: "<<engineFile<<"------------/"<<endl;
        engineID = pCreateEngine(engineID, trtModel, pMaskWeight);
        std::cout<<"The Engine Id is "<<engineID<<endl;
        cout<<"/------------Engined Created------------/"<<endl;

        cout<<"/------------Creating Output Buffer------------/"<<endl;
        int bufferNum = 0;
        EngineBuffer *bufferInfo;
        // int fsd_out_offset = -1;
        
        std::vector<int> classPos;
        std::vector<int> baseOffset(outClass.size());
        std::vector<int> writeNum   (outClass.size());

        if(pGetBuffer != NULL) {
            const char *sBufferType[ONNXTRT_MAX_BUFFERTYPE] = {"In:", "Out:", "Var:"};
	        sBufferType[10]="CPU_In:";
	        sBufferType[11]="CPU_Out:";      
            const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
            const char *sTensorFomat[6] = {"Linear", "CHW2", "HWC8", "CHW4", "CHW16", "CHW32"};
            pGetBuffer(engineID, &bufferInfo, &bufferNum, NULL);
            cout<<"Get Buffer Num is "<<bufferNum<<endl;

            for(int i = 0; i < outClass.size(); ++i) {
                const char *curClass = outClass[i].c_str();
                for(int j = 0; j < bufferNum; ++j) {
                    if (NULL != strcasestr(bufferInfo[j].name, curClass)) {
                        classPos.push_back(j - 1);
                        cout<<"find "<<curClass<<" as: "<<bufferInfo[j].name<<endl;
                        break;
                    } 

                    if(j == bufferNum - 1){
                        cout<<"ERROR: Cannot find "<<curClass<<". Return."<<endl;
                        return -1;
                    }  
                }
            }

            int accOffset = 0;

            for (int i = 0; i < bufferNum; ++i) {
                //Print out buffer information
                cout<<"Buf["<<i<<"] "<<sBufferType[bufferInfo[i].nBufferType]<<bufferInfo[i].name<<" "<<
                batchSize<<"x["<<bufferInfo[i].d[0]<<", "<<bufferInfo[i].d[1]<<", "<<bufferInfo[i].d[2]<<"], "
                <<sDataType[bufferInfo[i].nDataType]<<endl;

                if (bufferInfo[i].nBufferType == 1 || bufferInfo[i].nBufferType == 11) { // Output buffer
                    int outSize  = sizeof(float) * batchSize; // <= trt MaxBatch
                    int baseSize = batchSize;
                    for( int j = 0; j < bufferInfo[i].nDims; ++j) {
                        outSize  *= bufferInfo[i].d[j];
                        baseSize *= bufferInfo[i].d[j];
                    }   

                    accOffset += baseSize;

                    for(int j = 0; j < outClass.size(); ++j) {
                        if(classPos[j] == i)    baseOffset[j] = accOffset; 
                        if(classPos[j] == i - 1)writeNum[j]   = baseSize;
                    }
                        

                    outputSize += std::min(bufferInfo[i].nBufferSize, outSize) ;
                }
            }

            for(int j = 0; j < outClass.size(); ++j)
                cout<<"The "<<outClass[j]<<" offset is "<<baseOffset[j]
                <<"; write number is "<<writeNum[j]<<endl;
                

            cout<<"The output size is "<<outputSize<<endl;

        }else if (outputSize == 0) { // no outputSize set, get from environment
            char *val = getenv("TRT_OUTSIZE");
            if (NULL != val) {
                outputSize = atoi(val);
                printf("getenv TRT_OUTSIZE=%d\n", outputSize);
            }
        }
        
        cout<<"Creating CPU Output buffer, the output size is "<<outputSize<<endl;
        float *pOutData = new float[outputSize / sizeof(float) * batchSize];

        int inputTypeCopy = inputType;

        if(inputFile.size() == 0) {
            cout<<"ERROR: The txt list for images is not set. Retrun."<<endl;
            return -1;
        }

        //Reading data.
        cout<<"/------------Start reading images name from txt list------------/"<<endl;

        for(int nL = 0; nL < mTxtList.size() / batchSize; ++nL) {

            //Read All image path from txt list.
            std::vector<std::vector<std::string>> mImageListVector; //Batch - Image
            for(int nB = 0; nB < batchSize; ++nB) {
                int posInTxtList = nB * initLineCounter + nL;
                std::vector<std::string> mImageListTmp;
                std::ifstream inFile(mTxtList[posInTxtList]);
                if (inFile.good()) {
                    std::string imgPath;
                    while (std::getline(inFile, imgPath)) 
                        if (imgPath.length() > 2) 
                            mImageListTmp.push_back(imgPath);
                    if(mImageListTmp.size() <= 0) {
                        cout<<"ERROR: "<<mTxtList[posInTxtList]<<" doesn't contain any input image paths."<<endl;
                        return -1;
                    } else {
                        cout<<"The image number of "<<mTxtList[posInTxtList]<<" is "<<mImageListTmp.size()<<endl;
                    }
                } else{
                    cout<<"ERROR: Open "<<mTxtList[posInTxtList]<<" failed"<<endl;
                }

                mImageListVector.push_back(mImageListTmp);
            }

            for(int nN = 1; nN < batchSize; ++nN){
                if(mImageListVector[nN].size() != mImageListVector[nN - 1].size()) {
                    cout<<"ERROR: Image number read from "<< nN<<"th and "<< nN-1<<"th list are not equal.Return."<<endl;
                    return -1;
                }
            }

            int nImageNum = mImageListVector[0].size();

            cout<<"/------------Set output Path------------/"<<endl;
            std::string curOutPath;
            //creat folder for output
            if(loopMode) {
                std::string::size_type iPos = mTxtList[nL].find_last_of('/') + 1;
                std::string fName = mTxtList[nL].substr(iPos, mTxtList[nL].length() - iPos);
                std::string name = fName.substr(0, fName.rfind("."));

                curOutPath = outPath + "/" + name;
                // stringstream tmpStringStream;
                // std::string  tmpString;
                // tmpStringStream << (nL + 1);
                // tmpStringStream >> tmpString;
                
                // curOutPath = outPath + "/clip_" + tmpString;

                buildFolder(curOutPath);
            } else{
                curOutPath = outPath;
            }

            cout<<"/------------Start reading images------------/"<<endl;
            cout<<"There are "<<nImageNum<<" images in the current lists."<<endl;

            for (int nI = 0; nI < nImageNum; nI++) {

                inputType =  (sequenceMode && loopMode && (nI !=0)) ? (inputTypeCopy | ONNXTRT_CONVLSTM_MEMORY) : inputTypeCopy;

                cout<<nI<<": "<<inputType<<endl;

                int   imageWidth   = 0;
                int   imageHeight  = 0;
                int   imageChannel = 0;
                float imageScale   = 1.f;

                char          *inputStream = NULL;
                unsigned char *rgbBuffer   = NULL;

                std::vector<std::string> inputFileNameVector;
                
                
                for(int nB = 0; nB < batchSize; ++nB) {
                    inputFileNameVector.push_back(mImageListVector[nB][nI]); 
                    //Check The format of files
                    if (!inputFileNameVector[nB].empty()) {
                        if(inputFileNameVector[nB][inputFileNameVector[nB].size() - 1] == '\r') 
                            inputFileNameVector[nB].erase(inputFileNameVector[nB].size() - 1);
                    }else {
                        cout<<"ERROR: the input image name is empty. Return."<<endl;
                        return -1;
                    }
                }
                
                // if (strncmp("ConvLSTM:", inputFileNameVector[0].c_str(), 9) == 0) {
                //     if (strstr(inputFileNameVector[0].c_str(), "Video")) {
                //         cout<<"Note: Set ConvLSTM video flag"<<endl;
                //         inputType = inputTypeCopy | ONNXTRT_CONVLSTM_MEMORY;
                //         cout<<"The input type is "<<inputType<<endl;
                //     } else {
                //         cout<<"Note: Set ConvLSTM image flag"<<endl;
                //         // inputType = inputTypeCopy & ~ONNXTRT_CONVLSTM_MEMORY;
                //         inputType = inputTypeCopy;
                //         cout<<"The input type is "<<inputType<<endl;
                //     }

                //     continue;
                // }  
                    
                // cout<<inputFileNameVector.size()<<endl;
                int nameLength = inputFileNameVector[0].size() - 4;

                if (checkValidImageFormatImage(inputFileNameVector[0], nameLength)) {
                    if(!isLidarModel && !dataPPed) {
                        cout<<"Reading image using std"<<endl;
                        ret = readImageTo12Channel(inputFileNameVector, rgbBuffer, imageWidth, imageHeight,
                                                    imageChannel, inputStream, imageScale, inputTypeCopy, modelType);
                        if(inputTypeCopy == 3 || inputTypeCopy == 2) {
                            size_t byteSize = (inputTypeCopy == 2) ? sizeof(float) : sizeof(char); 
                            int copySize = batchSize * imageChannel * imageWidth * imageHeight;    
                            cout<<"copy size is "<<  copySize <<endl;
                            void* gpuPtr = pAllocateSpaceGPU(byteSize, copySize);
                            pMemcpyHost2DeviceGPU(gpuPtr, (void*)inputStream, byteSize, copySize);
                            delete[] inputStream;
                            inputStream = (char*) gpuPtr;
                            printf("Note:Already copied CPU buffer to GPU.\n");
                        }
                        if (ret < 0) {
                            cout<<"ERROR: "<<nI<<"th Image in the list is empty. Return."<<endl;
                            // return -1;
                            break;
                        }else {
                            cout<<"Read ";
                            
                            for(auto itFile = inputFileNameVector.begin(); itFile != inputFileNameVector.end(); itFile++)
                                cout<<*itFile<<" finish"<<endl;

                            
                        }
                    } else {
                        cout<<"ERROR: Please Check the Following two conditions: \n"
                            <<"1. Lidar Model doesn't support image input, are you using image format for lidar model?\n"
                            <<"2. The images who are already preprocessed have to be input as bin file, check the the format for preprocessed input."
                            <<endl;
                        // return -1;
                        break;
                    }
                        
                } else if (checkValidImageFormatBin(inputFileNameVector[0], nameLength)){

                    if (batchSize != 1) {
                        cout<<"ERROR: bin file doesn't support multi bathch input."<<endl;
                        return -1;
                    }

                    if(dataPPed && !isLidarModel) {
                        cout<<"Reading Bin file."<<endl;
                        ret = readBinFile(inputFileNameVector[0].c_str(), inputStream);
                        if (ret < 0) {
                            cout<<"ERROR: "<<nI<<"th Image in the list is empty. Return."<<endl;
                            // return -1;
                            break;
                        }else {
                            cout<<"Read ";
                            
                            for(auto itFile = inputFileNameVector.begin(); itFile != inputFileNameVector.end(); itFile++)
                                cout<<*itFile<<" finish"<<endl;
                        }
                    }else if(isLidarModel && !needGpuProcess) {
                        
                        ret = voxelGenerator(inputFileNameVector[0].c_str(), inputStream, dicName, scale, saveVoxel);
                        if (ret < 0) {
                        cout<<"ERROR: "<<nI<<"th Image in the list is empty. continue."<<endl;
                        continue; 
                        }else {
                            cout<<"Read ";
                            
                            for(auto itFile = inputFileNameVector.begin(); itFile != inputFileNameVector.end(); itFile++)
                                cout<<*itFile<<" finish"<<endl;
                        }

                    } else if(isLidarModel && needGpuProcess){
                        cout<<"Reading Bin file."<<endl;
                        ret = readBinFile(inputFileNameVector[0].c_str(), inputStream);
                        if (ret < 0) {
                            cout<<"ERROR: "<<nI<<"th Image in the list is empty. Return."<<endl;     
                            break;
                        }else {
                            lidarInput.pointCloud = (void*) inputStream;
                            lidarInput.pointNum   = ret / sizeof(PointXYZI);
                            inputStream = reinterpret_cast<char*>(&lidarInput);   
                            cout<<"Read ";
                            
                            for(auto itFile = inputFileNameVector.begin(); itFile != inputFileNameVector.end(); itFile++)
                                cout<<*itFile<<" finish"<<endl;
                        }
                    } else {
                            cout<<"ERROR: Please Check the Following two conditions: \n"
                            <<"1. If the image input is not preprocessed you have to input a image format, not a bin file.\n"
                            <<"2. If the bin file is raw lidar data, u need to specify the type of model using -t (ex. -t lidar)."
                            <<endl;
                
                            break;
                    }
                    
                    
                } else {
                    cout<<"ERROR: the format of "<<nI<<"th Image in the list is not support. Return."<<endl;
                    break;
                }

                //Warmup for ConvLSTM
                if(sequenceMode && nI == 0 && nL == 0) {
                    cout<<"Note: Warming up for ConvLSTM model"<<endl;
                    pRunEngine(engineID, batchSize, inputStream, inputTypeCopy, (char *)pOutData, outType);
                } 
                    
                

                for (int i = 0; i < 1; ++i) {
                    auto t_start = std::chrono::high_resolution_clock::now();
                    // Call tensorRT from C interface.
                    ret = pRunEngine(engineID, batchSize, inputStream, inputType,
                                    (char *)pOutData, outType);

                    float ms = std::chrono::duration<float, std::milli>(
                                    std::chrono::high_resolution_clock::now() - t_start)
                                    .count();
                    printf("RunEngine ret = %d , time = %f ms\n", ret, ms);
                }

                if(pGetBuffer != NULL && curOutPath.c_str() != NULL)
                    for (int nOutClass = 0; nOutClass < outClass.size(); ++nOutClass) {
                        if(isNpy)
                            writResultNumpy(pOutData, curOutPath, inputFileNameVector[0], outClass[nOutClass], 
                                            imageWidth, imageHeight, baseOffset[nOutClass], writeNum[nOutClass]);
                        else
                            writeResultBin(pOutData, curOutPath, inputFileNameVector[0], outClass[nOutClass], 
                                            imageWidth, imageHeight, baseOffset[nOutClass], writeNum[nOutClass]);
                    }
                        

                if (NULL != inputStream && inputTypeCopy != 3){
                    if(isLidarModel && needGpuProcess) {
                        delete[] lidarInput.pointCloud;
                        lidarInput.pointCloud = NULL;
                        lidarInput.pointNum   = 0;
                    } else {
                        delete[] inputStream;
                        inputStream = NULL;
                    }
                    
                }else if (NULL != inputStream && inputType == 3) {
                    pFreeSpaceGPU((void*)inputStream);
                }
            }
        }
        

        if (NULL != pOutData) delete[] pOutData;

        pDestoryEngine(engineID);
    } else {
        std::cout<<"Cannot load trt inference functions from "<<pSoName<<std::endl;
    }



}

