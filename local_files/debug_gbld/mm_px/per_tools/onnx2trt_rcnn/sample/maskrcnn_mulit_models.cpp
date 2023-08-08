/*
 * Copyright (c) 2018, Xpeng Motor. All rights reserved.
 * [caizw@20181130] Add openCV to read encoded images( readImageByOpenCV ) ,
 * openmp to run  mulit-models parallel example:
   bin/maskrcnn_mulit_models -s bin/libonnxtrt.so
    -e model_final_608x960b1.trt -i output6_960x608_134544.645.ppm
    -o output_final_960x608_mulit.ppm -m masknetweight.bin
    -e LLD_604x960fp32_MKZ.trt -i input_960_604.ppm
    -o output_LLD_960x604_mulit.ppm
 * MaskNet need weight "masknetweight.bin"
 */

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>  // For ::getopt
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
#include <fcntl.h>  // For ::open
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>
#include "cnpy.h"
#include "onnxtrt.h"
#include "string.h"

PTCreateEngine pCreateEngine;
PTRunEngine pRunEngine;
PTDestoryEngine pDestoryEngine;
PTGetBufferOfEngine pGetBuffer;

// return ms
double getTimeDiff(struct timeval *tm) {
  struct timeval lasttm = *tm;
  gettimeofday(tm, NULL);
  return (tm->tv_sec - lasttm.tv_sec) * 1000.f +
         (tm->tv_usec - lasttm.tv_usec) / 1000.f;
}

int getPreprocesParams(int &modelType, int &im_w, int &im_h, int *idxRGB,
                       float *Scales, float *Means) {
  // check for Width & Height
  char *val = getenv("TRT_IW");
  if (NULL != val) {
    im_w = atoi(val);
    printf("getenv TRT_IW=%d\n", im_w);
  }
  val = getenv("TRT_IH");
  if (NULL != val) {
    im_h = atoi(val);
    printf("getenv TRT_IH=%d\n", im_h);
  }

  val = getenv("TRT_BGR");
  if (NULL != val) {
    int is_BGR = atoi(val);
    printf("getenv TRT_BGR=%d\n", is_BGR);
    if (is_BGR) {
      idxRGB[0] = 2;
      idxRGB[2] = 0;
    }
  }

  if (1 == modelType) {
    Means[0] = 102.9801f;  // R
    Means[1] = 115.9465f;  // G
    Means[2] = 122.7717f;  // B
  }
  if (3 <= modelType) {
    Scales[0] = 255.f;  // R
    Scales[1] = 255.f;  // G
    Scales[2] = 255.f;  // B
  }

  return 0;
}

size_t ReadBinFile(const char *filename, char *&databuffer) {
  size_t size{0};
  std::ifstream file(filename, std::ifstream::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    databuffer = new char[size];
    // assert(databuffer);
    file.read(databuffer, size);
    file.close();
  }
  return size;
}

#ifndef USE_CV  // ppm only
int readImageByOpenCV(const char *filename, unsigned char *&rgbbuffer,
                      int &im_w, int &im_h, char *&databuffer,
                      int modelType = 1) {
  std::ifstream infile(filename, std::ifstream::binary);
  if (!infile.good()) return -1;

  char magic[8], w[256], h[8], max[8];
  infile.read(magic, 3);
  if (!('P' == magic[0] && ('5' == magic[1] || '6' == magic[1]))) {
    return -1;
  }

  infile >> w;
  if (w[0] == '#') {
    infile.getline(w, 255);
    infile >> w;
  }
  infile >> h >> max;
  im_h = atoi(h);
  im_w = atoi(w);

  int idxRGB[3] = {0, 1, 2};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  getPreprocesParams(modelType, im_w, im_h, idxRGB, Scales, Means);

  int imgsize = im_w * im_h;
  int size = imgsize;
  if (magic[1] == '6') size *= 3;
  // printf("Read magic=P%c w=%s, h=%s, size=%d modelType=%d\n", magic[1],w,h,
  // size, modelType);

  float *floatbuffer = new float[size];
  rgbbuffer = new unsigned char[size];
  infile.seekg(1, infile.cur);
  infile.read((char *)rgbbuffer, size);

  for (int i = 0; i < imgsize; i++) {
    floatbuffer[i] =
        (rgbbuffer[i * 3 + idxRGB[0]] - Means[0]) / Scales[0];  // R
    floatbuffer[i + imgsize] =
        (rgbbuffer[i * 3 + idxRGB[1]] - Means[1]) / Scales[1];  // G
    floatbuffer[i + imgsize * 2] =
        (rgbbuffer[i * 3 + idxRGB[2]] - Means[2]) / Scales[2];  // B
  }

  databuffer = (char *)floatbuffer;
  // delete[] rgbbuffer;

  return 0;
}
#else

#include <opencv2/opencv.hpp>
int readImageByOpenCV(const char *filename, unsigned char *&rgbbuffer,
                      int &im_w, int &im_h, char *&databuffer,
                      int modelType = 1) {
  cv::Mat mat_rgb = cv::imread(filename);
  if (!mat_rgb.data) {
    fprintf(stderr, "readImageByOpenCV read image fail: %s\n", filename);
    return -1;
  }
  im_w = mat_rgb.cols;
  im_h = mat_rgb.rows;

  int idxRGB[3] = {2, 1, 0};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  getPreprocesParams(modelType, im_w, im_h, idxRGB, Scales, Means);

  int imgsize = im_w * im_h;
  float *floatbuffer = new float[imgsize * 3];
  rgbbuffer = new unsigned char[imgsize * 3];

  int k = 0;
  cv::MatIterator_<cv::Vec3b> it_im, itEnd_im;
  it_im = mat_rgb.begin<cv::Vec3b>();
  itEnd_im = mat_rgb.end<cv::Vec3b>();
  for (; it_im != itEnd_im && k < imgsize; it_im++, k++) {
    rgbbuffer[k * 3 + idxRGB[2]] = (*it_im)[2];  // B
    rgbbuffer[k * 3 + idxRGB[1]] = (*it_im)[1];  // G
    rgbbuffer[k * 3 + idxRGB[0]] = (*it_im)[0];  // R

    floatbuffer[k] = (rgbbuffer[k * 3 + 2] - Means[0]) / Scales[0];  // R
    floatbuffer[k + imgsize] =
        (rgbbuffer[k * 3 + 1] - Means[1]) / Scales[1];  // G
    floatbuffer[k + imgsize * 2] =
        (rgbbuffer[k * 3 + 0] - Means[2]) / Scales[2];  // B
  }
  databuffer = (char *)floatbuffer;
  printf("readImageByOpenCV %s size=%d\n", filename, k);

  return 0;
}
#endif

// process the result of MaskRCNN-Resnet50(Detectron)
typedef struct {
  int classID;     // id of class [1,class_num]
  float roiScore;  // score of roi [0,1]
  float x;         // rect & mask in scaled image(not the orign image)
  float y;
  float width;
  float height;
  float *pMask;            // raw data of mask
  std::vector<char> mask;  // mask fit the rect, 0: no mask
} TRoiMaskOut;
static int num_classes = 9;  // cfg.MODEL.NUM_CLASSES = 6
const int MaskSize = 28;
const int DETECTNUM = 100;
static int MaskBlockSize = 0;

void ParseMaskData(float *pOutData, std::vector<TRoiMaskOut> &roiMasks) {
  int postNMS = 300;
  float *tmpData = pOutData;
  // check for nms num (10 or 100 or 300)
  char *val = getenv("TRT_POSTNMS");
  if (NULL != val) {
    postNMS = atoi(val);
    printf("getenv TRT_POSTNMS=%d\n", postNMS);
  }
  val = getenv("TRT_CLASS");
  if (NULL != val) {
    num_classes = atoi(val);
    printf("getenv TRT_CLASS=%d\n", num_classes);
  }

  const int MaskOutSize[5] = {
      postNMS * 5, postNMS * num_classes * 4, postNMS * num_classes,
      DETECTNUM * (5 * 5 + 1),
      DETECTNUM * MaskSize * MaskSize *
          num_classes};  // rois/bbox_pred/cls_prob / nms rois / masks
  MaskBlockSize = MaskOutSize[0] + MaskOutSize[1] + MaskOutSize[2] +
                  MaskOutSize[3] + MaskOutSize[4];

  int roiNum_nms = 0;
  for (int outid = 0; outid < 5; outid++) {
    std::cerr << "Mask outData" << outid << "[" << MaskOutSize[outid] << "] = ";
    for (int eltIdx = 0; eltIdx < MaskOutSize[outid] && eltIdx < 8; ++eltIdx) {
      std::cerr << tmpData[eltIdx] << ", ";
    }
    std::cerr << std::endl;

    if (3 == outid) {  // nms rois
      for (int i = 0; i < 100; i++) {
        if (tmpData[5 * i] > 1e-5) {
          int classID = floor(tmpData[5 * i]);  // classID + 0.1f*roiScore
          float roiScore = (tmpData[5 * i] - classID) * 10;
          float left = tmpData[5 * i + 1];
          float top = tmpData[5 * i + 2];
          float right = tmpData[5 * i + 3];
          float bottom = tmpData[5 * i + 4];
          printf(
              "%d classID=%d roiScore=%f left=%f top=%f right=%f bottom=%f\n",
              i, classID, roiScore, left, top, right, bottom);
          roiNum_nms = i + 1;

          TRoiMaskOut tmpRoi = {classID,      roiScore,     left, top,
                                right - left, bottom - top, NULL};
          roiMasks.push_back(tmpRoi);
        } else {
          break;
        }
      }
    }

    if (4 == outid) {                                          // masks
      const int maskSize = MaskSize * MaskSize * num_classes;  // float(4byte)
      for (int i = 0; i < roiNum_nms; i++) {
        roiMasks[i].pMask = tmpData + i * maskSize;

        float *pData = roiMasks[i].pMask;
        float *pDataEnd = roiMasks[i].pMask + maskSize;
        printf("%d classID=%d  Mask={%f, %f, %f, ... %f, %f, %f }\n", i,
               roiMasks[i].classID, pData[0], pData[1], pData[2], pDataEnd[-3],
               pDataEnd[-2], pDataEnd[-1]);
      }
    }

    tmpData += MaskOutSize[outid];
  }
}

#ifndef USE_CV
int saveImageByOpenCV(std::string filename, unsigned char *imgbuffer, int im_w,
                      int im_h) {
  return -1;
}
#else
int saveImageByOpenCV(std::string filename, unsigned char *imgbuffer, int im_w,
                      int im_h) {
  cv::Mat bgrImg(im_h, im_w, CV_8UC3, imgbuffer);
  cv::Mat rgbImg(im_h, im_w, CV_8UC3);

  int from_to[] = {0, 2, 1, 1, 2, 0};  // From bgr to rgb
  cv::mixChannels(&bgrImg, 1, &rgbImg, 1, from_to, 3);

  cv::imwrite(filename, rgbImg);

  return 0;
}
#endif

// Simple code to draw line
inline void DrawLine(unsigned char *rgbbuffer, int im_w, int im_h,
                     const int color[], int x0, int y0, int x1, int y1) {
  if (x1 == x0) {
    if (y0 > y1) {
      int tmp = y1;
      y1 = y0;
      y0 = tmp;
    }
    for (int iy = y0; iy < y1; iy++) {
      int offset = (iy * im_w + x0) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = color[ch];
      }
    }
  } else {
    float scale = (float)(y1 - y0) / (x1 - x0);
    if (x0 > x1) {
      int tmp = x1;
      x1 = x0;
      x0 = tmp;
      y0 = y1;
    }

    for (int ix = x0; ix < x1; ix++) {
      int iy = (ix - x0) * scale + y0;
      int offset = (iy * im_w + ix) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = color[ch];
      }
    }
  }
}

void DrawMaskResult(std::vector<TRoiMaskOut> &roiMasks,
                    unsigned char *rgbbuffer, int im_w, int im_h,
                    const char *outfile) {
  const int im_size = im_w * im_h;
  const float im_scale = 960.0f / 960.0f;
  const int classColor[12][3] = {{0, 255, 0},   {31, 223, 0},  {255, 0, 0},
                                 {223, 31, 0},  {255, 224, 0}, {63, 63, 0},
                                 {255, 127, 0}, {127, 0, 64},  {0, 255, 127},
                                 {127, 255, 0}, {127, 0, 255}, {0, 127, 255}};
  for (unsigned int i = 0; i < roiMasks.size(); i++) {
    float ox = roiMasks[i].x * im_scale;
    float oy = roiMasks[i].y * im_scale;
    float w = roiMasks[i].width * im_scale;
    float h = roiMasks[i].height * im_scale;
    // float *pMask = roiMasks[i].pMask;
    int classID = roiMasks[i].classID;
    int colorID = classID - 1;
    printf("Draw classID=%d  rect={%f,%f,%f,%f}\n", classID, ox, oy, w, h);

    float scale_x = w / MaskSize;
    float scale_y = h / MaskSize;
    int tx = -1, ty = -1;  // pre point
    int tx0 = 0, ty0 = 0;  // first point
    // Draw Mask
    if (classID == 5) {
      float *pContour = roiMasks[0].pMask;
      for (int j = 0; j < (MaskSize * MaskSize); j++) {
        int val = (int)pContour[j];
        if (0 == val || -10 == val) break;

        if (val != -1) {
          int iy = (val / 60 - 0.5f) * scale_y / 2 + oy - 10;
          int ix = (val % 60 - 0.5f) * scale_x / 2 + ox - 10;
          iy = std::min(im_h - 1, std::max(0, iy));
          ix = std::min(im_w - 1, std::max(0, ix));

          if (tx >= 0) {
            printf(" ,%d(%d,%d)", val, iy, ix);
            DrawLine(rgbbuffer, im_w, im_h, classColor[colorID], tx, ty, ix,
                     iy);
          } else {
            printf("%u Contour %d(%d,%d)", i, val, iy, ix);
            tx0 = ix;
            ty0 = iy;
          }
          tx = ix;
          ty = iy;
        } else {
          DrawLine(rgbbuffer, im_w, im_h, classColor[colorID], tx, ty, tx0,
                   ty0);
          tx = -1;
          ty = -1;
        }
      }
      printf(" ,EndContour\n");
      continue;
    }

    // Draw Keypoint
    if (classID != 5) {
      float *pKP = roiMasks[1].pMask + i * 10 * 2;  // 10x index&conf
      for (int j = 0; j < 8; j++) {
        int index = (int)pKP[(j % 7) * 2];
        float conf = pKP[(j % 7) * 2 + 1];
        int iy = (index / MaskSize + 0.5f) * scale_y + oy;
        int ix = (index % MaskSize + 0.5f) * scale_x + ox;
        printf("%u maxIdx %d[%d,%d] conf=%f\n", i, index, iy, ix, conf);
        if (conf < 1.4f) continue;

        iy = std::min(im_h - 1, std::max(0, iy));
        ix = std::min(im_w - 1, std::max(0, ix));

        if (j > 0) {
          DrawLine(rgbbuffer, im_w, im_h, classColor[colorID], tx, ty, ix, iy);
        }
        tx = ix;
        ty = iy;
      }
      continue;
    }

    // Draw box
    for (int x = 0; x < w; x++) {
      int offset = (int(oy) * im_w + int(ox + x)) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = classColor[colorID][ch];
        rgbbuffer[offset + int(h) * im_w * 3 + ch] = classColor[colorID][ch];
      }
    }
    for (int y = 0; y < h; y++) {
      int offset = (int(oy + y) * im_w + int(ox)) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = classColor[colorID][ch];
        rgbbuffer[offset + int(w) * 3 + ch] = classColor[colorID][ch];
      }
    }
  }

  if (NULL != outfile) {
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
  }
}
// process the output of LLD model---------------------------------------------
void DrawLLDResult(float *LLDoutput, unsigned char *rgbbuffer, int im_w,
                   int im_h, const char *outfile, int outsize) {
  const int im_size = im_w * im_h;

  if (NULL != outfile) {
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
  }

  unsigned int valNum = outsize / sizeof(float);
  if (valNum > 0 && NULL != outfile) {
    std::string filename = outfile;
    cnpy::npy_save(filename + ".npy", LLDoutput, {valNum});
  }
}

void print_usage() {
  cout << "TensorRT maskcnn sample" << endl;
  cout << "Usage: onnx2trt onnx_model.pb"
       << "\n\t[-o output.ppm]  (output image as ppm)"
       << "\n\t[-e engine_file.trt]  (test TensorRT engines, can "
          "be specified multiple times)"
       << "\n\t[-i input_data.bin]  (input datas, can be specified "
          "multiple times)"
       << "\n\t[-m modelType ( LLD | MaskNetWeight.bin | LaneNet | "
          "MODNet | LaneMOD | LLDMOD ) ]"
       << "\n\t[-s libonnxtrt.so] (path of libonnxtrt.so)"
       << "\n\t[-b max_batch_size (default 1)]"
       << "\n\t[-h] (show help)" << endl;
}

float percentile(float percentage, std::vector<float> &times) {
  int all = static_cast<int>(times.size());
  int exclude = static_cast<int>((1 - percentage / 100) * all);
  if (0 <= exclude && exclude <= all) {
    std::sort(times.begin(), times.end());
    float pctTime = times[all == exclude ? 0 : all - 1 - exclude];
    float totTime = 0;
    for (int i = 5; i < all - 5; i++)
      totTime += times[i];  // drop the fist & last 5 datas.

    printf(
        "TestAll %d range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< "
        "%.3fms\n",
        all, times[0], times[all - 1], totTime / (all - 10), times[all / 2],
        percentage, pctTime);
    return pctTime;
  }
  return std::numeric_limits<float>::infinity();
}

static int testRunEngine(int ch, int batch, char *inputData, int inputType,
                         char *outData, int outType) {
  if (NULL == pRunEngine) return -1;

  auto t_start = std::chrono::high_resolution_clock::now();

  int nRet = pRunEngine(ch, batch, inputData, inputType, outData, outType);

  float ms = std::chrono::duration<float, std::milli>(  // 0.4us
                 std::chrono::high_resolution_clock::now() - t_start)
                 .count();
  printf("CH[%d] time = %f\n", ch, ms);  // cout : 0.1ms, printf: 0.03ms

  return nRet;
}

#include <condition_variable>
#include <mutex>
static std::mutex mtx;
static std::condition_variable condvar;
bool ready = true;
float g_times[16], max_ms;
std::chrono::time_point<std::chrono::high_resolution_clock> g_start;
static int testRunEngine1(int ch, int batch, char *inputData, int inputType,
                          char *outData, int outType) {
  int nRet = 0;
  {
    std::unique_lock<std::mutex> lck(mtx);
    condvar.wait(lck);
  }
  while (ready) {
    nRet = pRunEngine(ch, batch, inputData, inputType, outData, outType);

    g_times[ch] = std::chrono::duration<float, std::milli>(  // 0.4us
                      std::chrono::high_resolution_clock::now() - g_start)
                      .count();
    printf("CH[%d] time = %f\n", ch,
           g_times[ch]);  // cout : 0.1ms, printf: 0.03ms
    max_ms = g_times[ch];
    {
      std::unique_lock<std::mutex> lck(mtx);
      condvar.wait(lck);
    }
  };

  return nRet;
}

// print the fist & last 64(PrintNum) data
void printOutput(int64_t eltCount, float *outputs) {
  const int PrintNum = 64;
  for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {
    std::cerr << outputs[eltIdx] << "\t, ";
  }
  if (eltCount > PrintNum) {
    std::cerr << " ... ";
    for (int64_t eltIdx = (eltCount - PrintNum); eltIdx < eltCount; ++eltIdx) {
      std::cerr << outputs[eltIdx] << "\t, ";
    }
  }

  std::cerr << std::endl;
}

// print the data of buffer, split by bufferInfo
void PrintOutBuffer(float *pBuf, int outType, int batchsize, int bufferNum,
                    EngineBuffer *bufferInfo) {
  int valNum = 0;
  if (0 == outType) {  // plan buffers, default
    for (int i = 0; i < bufferNum; i++) {
      for (int nb = 0; nb < batchsize; nb++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++)
          eltCount *= bufferInfo[i].d[d];
        printf("Out[%d].%d eltCount:%d Data:\n", i, nb, eltCount);
        printOutput(eltCount, pBuf + valNum);
        valNum += eltCount;
      }
    }
  } else if (2 == outType) {  // packed buffers by batchsize
    for (int nb = 0; nb < batchsize; nb++) {
      for (int i = 0; i < bufferNum; i++) {
        if (0 == bufferInfo[i].nBufferType) continue;

        int eltCount = 1;
        for (int d = 0; d < bufferInfo[i].nDims; d++)
          eltCount *= bufferInfo[i].d[d];
        printf("Out[%d].%d eltCount:%d Data:\n", i, nb, eltCount);
        printOutput(eltCount, pBuf + valNum);
        valNum += eltCount;
      }
    }
  }
}

typedef struct IMGInfo {
  int im_w = 0;
  int im_h = 0;
  unsigned char *rgbbuffer = NULL;
  char *inputStream;
  int modelType;  // 0: LLD, 1: MaskRCNN/FasterRCNN, 2: LaneNet, 3 MODNet, 4
                  // Lane & MOD, 5 LLDMOD
} IMGInfo;

int main(int argc, char *argv[]) {
  if (argc <= 1) print_usage();
  std::vector<std::string> engine_filenames;
  std::vector<std::string> maskweight_filenames;
  std::vector<std::string> input_filenames;
  std::vector<std::string> out_filenames;
  std::vector<int> batch_sizes;
  float fps_delay = 30;  // delay ms to get fixed FPS
  const char *pSoName = "./libonnxtrt.so";
  bool debug_output = false;
  int MAX_TEST = 100;
  int thread_type = 0;

  int arg = 0;
  while ((arg = ::getopt(argc, argv, "o:e:i:m:s:b:F:t:T:gh")) != -1) {
    switch (arg) {
      case 'o':
        if (optarg) {
          out_filenames.push_back(optarg);
          break;
        } else {
          cerr << "ERROR: -o flag requires argument" << endl;
          return -1;
        }
      case 'e':
        if (optarg) {
          engine_filenames.push_back(optarg);
          break;
        } else {
          cerr << "ERROR: -e flag requires argument" << endl;
          return -1;
        }
      case 'i':
        if (optarg) {
          input_filenames.push_back(optarg);
          break;
        } else {
          cerr << "ERROR: -i flag requires argument" << endl;
          return -1;
        }
      case 'm':
        if (optarg) {
          maskweight_filenames.push_back(optarg);
          break;
        } else {
          cerr << "ERROR: -m flag requires argument" << endl;
          return -1;
        }
      case 's':
        if (optarg) {
          pSoName = optarg;
          break;
        } else {
          cerr << "ERROR: -s flag requires argument" << endl;
          return -1;
        }
      case 'F':  // fixed FPS = 20;
        fps_delay = 1000.f / (atoi(optarg) + 0.01);
        break;
      case 't':  // default 100
        MAX_TEST = atoi(optarg);
        break;
      case 'T':  // thread_type 0, 1, 2
        thread_type = atoi(optarg);
        break;
      case 'b':
        if (optarg) {
          batch_sizes.push_back(atoll(optarg));
          break;
        } else {
          cerr << "ERROR: -b flag requires argument" << endl;
          return -1;
        }
      case 'g':
        debug_output = true;  // print output
        break;
      case 'h':
        print_usage();
        return 0;
    }
  }

  void *pLibs = dlopen(pSoName, RTLD_LAZY);
  if (NULL == pLibs) {
    printf("Can not open library %s\n", pSoName);
    return -1;
  }

  {
    pCreateEngine = (PTCreateEngine)dlsym(pLibs, "CreateEngine");
    pRunEngine = (PTRunEngine)dlsym(pLibs, "RunEngine");
    pDestoryEngine = (PTDestoryEngine)dlsym(pLibs, "DestoryEngine");
    pGetBuffer = (PTGetBufferOfEngine)dlsym(pLibs, "GetBufferOfEngine");
  }

  if (NULL != pCreateEngine && NULL != pRunEngine &&
      NULL != pDestoryEngine) {  // run models parallel
    unsigned int engine_num = engine_filenames.size();
    char *output[16] = {NULL};
    int outsizes[16];
    IMGInfo input[16];
    memset(input, 0, sizeof(input));
    memset(outsizes, 0, sizeof(outsizes));
    int bufferNum[16];
    EngineBuffer *bufferInfo[16];

    for (unsigned int ch = 0; ch < engine_num; ch++) {  // ID for multi-engines
      const char *pMaskWeight = maskweight_filenames[ch].c_str();
      if (0 == strncasecmp(pMaskWeight, "Mask", 4)) {
        input[ch].modelType = 1;
      }

      // default batchsize = 1
      if (batch_sizes.size() <= ch) batch_sizes.push_back(1);

      if (input_filenames.size() > ch) {
        std::string input_filename = input_filenames[ch];
        int ret = 0;
        if (!input_filename.empty()) {
          if (0 ==
              input_filename.compare(input_filename.size() - 4, 4, ".bin")) {
            size_t size =
                ReadBinFile(input_filename.c_str(), input[ch].inputStream);
            printf("ReadBinFile to inputStream size=%lu Bytes\n", size);
            if (size <= 0) ret = -1;
          } else {  // Using Opencv to read image format
            ret = readImageByOpenCV(input_filename.c_str(), input[ch].rgbbuffer,
                                    input[ch].im_w, input[ch].im_h,
                                    input[ch].inputStream, input[ch].modelType);
          }

          if (0 != ret) {
            fprintf(stderr, "readImage Read image fail: %s\n",
                    input_filename.c_str());
          }
        }
      }

      pCreateEngine(ch, engine_filenames[ch].c_str(), pMaskWeight);

      int &outputSize = outsizes[ch];
      if (NULL != pGetBuffer) {  // Get from GetBufferOfEngine.
        const char *sBufferType[2] = {"In:", "Out:"};
        const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
        pGetBuffer(ch, &bufferInfo[ch], &bufferNum[ch], NULL);
        printf("GetBuffer num = %d\n", bufferNum[ch]);
        for (int i = 0; i < bufferNum[ch]; i++) {
          EngineBuffer &bufInfo = bufferInfo[ch][i];
          printf("Buf[%d] %s %dx[%d,%d,%d], %s\n", i,
                 sBufferType[bufInfo.nBufferType], batch_sizes[ch],
                 bufInfo.d[0], bufInfo.d[1], bufInfo.d[2],
                 sDataType[bufInfo.nDataType]);
          if (1 == bufInfo.nBufferType) {                   // Output buffer
            int outSize = sizeof(float) * batch_sizes[ch];  // <= trt MaxBatch
            for (int j = 0; j < bufInfo.nDims; j++) outSize *= bufInfo.d[j];
            outputSize += std::min(bufInfo.nBufferSize, outSize);
          }
        }
        printf("Tot outputSize=%d B\n", outputSize);
      } else if (0 == outputSize) {  // no outputSize set, get from environment
        char *val = getenv("TRT_OUTSIZE");
        if (NULL != val) {
          outputSize = atoi(val);
          printf("getenv TRT_OUTSIZE=%d\n", outputSize);
        }
      }

      pRunEngine(0, batch_sizes[ch], input[ch].inputStream, 0, nullptr,
                 input[ch].modelType);

      output[ch] = new char[outputSize];
    }

    const float pct = 90.0f;
    std::vector<float> times(MAX_TEST);
    xpilot_os::Thread threads[16];

    if (0 == thread_type) {
      for (int nT = 0; nT < MAX_TEST; nT++) {
        auto t_start = std::chrono::high_resolution_clock::now();

        for (unsigned int ch = 0; ch < engine_num; ch++) {
          threads[ch] = xpilot_os::Thread(testRunEngine, ch, batch_sizes[ch],
                                    input[ch].inputStream, 0, output[ch],
                                    input[ch].modelType);
        }
        for (unsigned int ch = 0; ch < engine_num; ch++) {
          threads[ch].join();
        }

        float ms = std::chrono::duration<float, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_start)
                       .count();
        printf("Tot time = %f\n", ms);
        times[nT] = ms;
      }
    } else if (1 == thread_type || 11 == thread_type) {
      for (unsigned int ch = 0; ch < engine_num; ch++) {
        threads[ch] = xpilot_os::Thread(testRunEngine1, ch, batch_sizes[ch],
                                  input[ch].inputStream, 0, output[ch],
                                  input[ch].modelType);
      }
      usleep(1000);
      for (int nT = 0; nT < MAX_TEST; nT++) {
        memset(g_times, 0, sizeof(g_times));
        g_start = std::chrono::high_resolution_clock::now();
        condvar.notify_all();
        for (unsigned int ch = 0; ch < engine_num; ch++) {
          for (int i = 0; i < 100; i++) {
            if (g_times[ch] < 0.1f)
              usleep(fps_delay * 1000 / 100);
            else if (11 == thread_type) {
              ch = engine_num;
              break;
            }
          }
        }
        // printf("Tot time = %f\n", max_ms);  // cout : 0.1ms, printf: 0.03ms
        float ms = std::chrono::duration<float, std::milli>(
                       std::chrono::high_resolution_clock::now() - g_start)
                       .count();
        printf("Tot time = %f\n", ms);
        times[nT] = ms;
      }
      ready = false;
      condvar.notify_all();
      for (unsigned int ch = 0; ch < engine_num; ch++) {
        threads[ch].join();
      }
    } else {  // no Thread, =2 or =-1
      EngineBuffer *pBufferInfo[16];
      int bufNum[16];
      for (int nT = 0; nT < MAX_TEST; nT++) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int outType = (2 == thread_type) ? -1 : -3;
        if (2 == thread_type || 3 == thread_type) {
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            pRunEngine(ch, batch_sizes[ch], input[ch].inputStream, 0,
                       output[ch], outType);
          }
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            pGetBuffer(ch, &pBufferInfo[ch], &bufNum[ch], NULL);
            float ms = std::chrono::duration<float, std::milli>(
                           std::chrono::high_resolution_clock::now() - t_start)
                           .count();
            printf("CH.%d time = %f\n", ch, ms);
          }
        } else {  // -1
          for (unsigned int ch = 0; ch < engine_num; ch++) {
            pRunEngine(ch, batch_sizes[ch], input[ch].inputStream, 0,
                       output[ch], 0);
            float ms = std::chrono::duration<float, std::milli>(
                           std::chrono::high_resolution_clock::now() - t_start)
                           .count();
            printf("CH.%d time = %f\n", ch, ms);
          }
        }
        float ms = std::chrono::duration<float, std::milli>(
                       std::chrono::high_resolution_clock::now() - t_start)
                       .count();
        printf("Tot time = %f\n", ms);
        times[nT] = ms;
      }
    }

    percentile(pct, times);

    for (unsigned int ch = 0; ch < engine_num; ch++) {  // engine_num
      int modelType = input[ch].modelType;
      float *pOutData = (float *)output[ch];
      if (debug_output) {
        PrintOutBuffer(pOutData, modelType, batch_sizes[ch], bufferNum[ch],
                       bufferInfo[ch]);
      }

      if (out_filenames.size() > ch) {
        std::string outfile = out_filenames[ch];

        for (int nb = 0; nb < batch_sizes[ch];
             nb++) {  // process for every batch
          unsigned char *rgbbuffer = input[ch].rgbbuffer;
          int im_w = input[ch].im_w;
          int im_h = input[ch].im_h;

          if (1 == modelType) {
            std::vector<TRoiMaskOut> roiMasks;
            ParseMaskData(pOutData + nb * MaskBlockSize, roiMasks);
            if (NULL != rgbbuffer) {
              DrawMaskResult(roiMasks, rgbbuffer, im_w, im_h, outfile.c_str());
            }
          } else {
            DrawLLDResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str(),
                          outsizes[ch]);
          }

          if (0 == out_filenames[ch].compare(out_filenames[ch].size() - 4, 4,
                                             ".ppm")) {
            std::ofstream file(out_filenames[ch].c_str(),
                               std::ios::out | std::ios::binary);
            file << "P6\n"
                 << input[ch].im_w << " " << input[ch].im_h << "\n255\n";
            file.write((char *)input[ch].rgbbuffer,
                       (input[ch].im_w * input[ch].im_h) * 3);
            file.close();
          } else {  // Using Opencv to save other format
            saveImageByOpenCV(out_filenames[ch], input[ch].rgbbuffer,
                              input[ch].im_w, input[ch].im_h);
          }
        }
      }
    }

    for (unsigned int ch = 0; ch < engine_num; ch++) {
      pDestoryEngine(ch);
      if (NULL != input[ch].inputStream) {
        delete[] input[ch].inputStream;
      }
      if (NULL != output[ch]) {
        delete[] output[ch];
      }
    }
  } else {
    printf("Can not load library %s\n", pSoName);
  }

  dlclose(pLibs);

  return 0;
}
