#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "onnxtrt.h"

// return s
double getTimeDiff(struct timeval *tm) {
  struct timeval lasttm = *tm;
  gettimeofday(tm, NULL);
  return (tm->tv_sec - lasttm.tv_sec) + (tm->tv_usec - lasttm.tv_usec) * 1.0e-6;
}

void Run_processimg(int testNum) {
  struct timeval tm;
  printf("%s:%d\n", __FUNCTION__, __LINE__);
  fflush(stdout);

  // Call tensorRT from C interface. Else called from Python CDLL
  int ret =
      CreateEngine(0, "/home/nvidia/res_net_14out.trt", "MaskNetWeight.bin");
  printf("CreateEngine = %d\n", ret);

  getTimeDiff(&tm);
  for (int i = 0; i < testNum; i++) {
    // Call tensorRT from C interface. Else called from Python CDLL
    ret = RunEngine(0, 1, NULL, 0, NULL, 0);
  }
  printf("C host Call ret=%d time=%f s\n", ret, getTimeDiff(&tm) / testNum);

  DestoryEngine(0);
}

int main(int argc, char *argv[]) {
  for (int nT = 0; nT < 2; nT++) {
    Run_processimg(2);
  }
  return 0;
}
