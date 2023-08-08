#include <cuda.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>
#include "common.h"
using namespace std;


int     parseSocketPath(int argc, char *argv[], vector<string>& socketPath);
int     parseServiceNumber(int argc, char *argv[], int& serviceNum);
int     parseIterationNumber(int argc, char *argv[], int& iterations);
void    printUsage(void);
int     NUMTRIALS = 10;

CUresult
cudaDeviceCreate()
{
    CUdevice device;
    CUresult status = CUDA_SUCCESS;

    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuDeviceGet(&device, 0))) {
        printf("failed to get CUDA device\n");
        return status;
    }

    int major = 0, minor = 0;
    char deviceName[256];
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDeviceGetName(deviceName, 256, device);
    printf("CUDA Producer on GPU Device %d: \"%s\" with compute capability %d.%d\n\n", device, deviceName, major, minor);

    if (major < 6)
    {
        printf("EGLStream_CUDA_CrossGPU requires SM 6.0 or higher arch GPU.  Exiting...\n");
        exit(2); // EXIT_WAIVED
    }

    return status;
}

int parseSocketPath(int argc, char *argv[], vector<string>& socketPath)
{
    int i;

    for(i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-sp") == 0) {
            ++i;
            string tmp(argv[i]);
            socketPath.push_back(tmp);
        }
    }

    return 0;

}

int parseServiceNumber(int argc, char *argv[], int& serviceNum)
{
    int i;

    for(i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-sn") == 0) {
            ++i;
            serviceNum = atoi(argv[i]);
        }
    }

    return serviceNum;

}

int parseIterationNumber(int argc, char *argv[], int& iterations)
{
    int i;

    for(i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0) {
            ++i;
            iterations = atoi(argv[i]);
        }
    }

    return iterations;

}


void printUsage(void)
{
    printf("Usage:\n");
    printf("----Inference Service----\n");
    printf("  -sp          The socket path need to be created.\n");
    printf("  -sn          The number of services need to be created.\n");
    printf("  -n           Exit after running n trials. Set to 10 by default\n");

    printf("----Client----\n");
    printf("  -sn          The number of services need to be created.\n");
    printf("  -n           Exit after running n trials. Set to 10 by default\n");
}
