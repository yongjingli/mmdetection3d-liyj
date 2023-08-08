// example: bin/maskrcnn_cpp model_320x576b2.trt bin/libonnxtrt.so
// output6_576x320_134544.645.ppm gpu_0/masknetweight.bin bin/maskrcnn_cpp
// LLD_604x960fp32_MKZ.trt bin/libonnxtrt.so 171206_LLD_960x604.ppm MaskNet need
// weight "gpu_0/masknetweight.bin" , test data "gpu_0/_[mask]_roi_feat.bin"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "cnpy.h"
#include "onnxtrt.h"

PTCreateEngine pCreateEngine;
PTRunEngine pRunEngine;
PTDestoryEngine pDestoryEngine;
PTGetBufferOfEngine pGetBuffer;

static bool testInMemory = false;
static int batchsize = 1;
static int outputSize = 0;

#ifdef USE_CV
// draw contours with OpenCV, added by Chengzhang
#include <opencv2/imgcodecs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
void DrawContours(void *contour_buffer) {
  float *tmpContour = (float *)contour_buffer;
  std::vector<std::vector<int>> BoundaryPoints;
  std::vector<int> contour;
  int iii = 0;
  unsigned int c_idx = 0;
  // std::string input_name = "../Mask_all_th_0.25.pgm";

  // std::string input_name =
  // "/home/jetson/Chengzhang_xpilot/xpilot-feature-trtexec_plugin-caizw_0620/universe/workflow/tensorrt/tensorrt_sdk/maskrcnn_convert/Mask_all_th_0.25.pgm";
  std::string input_name = "Mask_all_th_0.25.pgm";

  std::string output_name = "draw_mask.png";
  cv::Mat imgDisplay = cv::imread(input_name, 1);
  int display_col = imgDisplay.cols;

  while (tmpContour[iii] != -10) {
    if (tmpContour[iii] != -1) {
      contour.push_back((int)tmpContour[iii]);
    } else {
      BoundaryPoints.push_back(contour);
      std::cout << "push_back contours!" << std::endl;
      contour.clear();
    }
    iii++;
  }

  std::cout << "number of contours " << BoundaryPoints.size() << std::endl;

  for (unsigned int Ind = 0; Ind < BoundaryPoints.size(); Ind++) {
    contour = BoundaryPoints[Ind];
    for (c_idx = 0; c_idx < contour.size() - 1;
         ++c_idx)  // didn't draw the last point
    {
      cv::line(
          imgDisplay,
          cv::Point(contour[c_idx] % display_col, contour[c_idx] / display_col),
          cv::Point(contour[c_idx + 1] % display_col,
                    contour[c_idx + 1] / display_col),
          cv::Scalar(0, 255, 0), 1);
    }
    cv::line(imgDisplay,
             cv::Point(contour[c_idx] % display_col,
                       contour[c_idx] / display_col),  // draw last point?
             cv::Point(contour[0] % display_col, contour[0] / display_col),
             cv::Scalar(0, 255, 0), 1);
  }

  cv::imwrite(output_name, imgDisplay);
}
#endif

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

#ifdef USE_softISP
#include "softISP.cuh"
#define WIDTH 1828
#define TOP_OFFSET 5
#define BOTTOM_OFFSET 3
#define RAW_HEIGHT 956
#define HEIGHT 948
#define BITOFFSET 0
#endif

size_t ReadBinFile(const char *filename, unsigned char *&rgbbuffer, int &im_w,
                   int &im_h, char *&databuffer, int modelType = 1) {
  size_t size{0};
  std::ifstream file(filename, std::ifstream::binary);
  if (!file.good()) return -1;

  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  databuffer = new char[size];
  // assert(databuffer);
  file.read(databuffer, size);
  file.close();

  int idxRGB[3] = {0, 1, 2};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  getPreprocesParams(modelType, im_w, im_h, idxRGB, Scales, Means);

  int imgsize = im_w * im_h;
  if (0 >= imgsize) {
    printf("Error! ReadBinFile need TRT_IW & TRT_IH\n");
    return -1;
  } else {
    // convert float RGB plan data to uint8 BGR packaged data
    float *floatbuffer = (float *)databuffer;
    rgbbuffer = new unsigned char[imgsize * 3];

#ifdef USE_softISP
    if (modelType == 7 || modelType == 8) {
      unsigned short *srcImg =
          (unsigned short *)databuffer + TOP_OFFSET * WIDTH;
      TestToneMapping(WIDTH, HEIGHT, WIDTH, srcImg, rgbbuffer, floatbuffer, 0.5,
                      1);
      int pre_w = WIDTH / 4, pre_h = HEIGHT / 4;
      if (im_h > pre_h) {
        for (int c = 3 - 1; c >= 0; c--) {
          for (int i = pre_h - 1; i >= 0; i--) {
            memmove(floatbuffer + (i + im_h * c) * im_w,
                    floatbuffer + (i + pre_h * c) * pre_w, pre_w * 4);
            memset(floatbuffer + (i + im_h * c) * im_w + pre_w, 0,
                   (im_w - pre_w) * 4);
          }
          memset(floatbuffer + (im_h * c + pre_h) * im_w, 0,
                 (im_h - pre_h) * im_w * 4);
        }
      }
    }
#endif

    for (int i = 0; i < imgsize; i++) {
      rgbbuffer[i * 3 + idxRGB[0]] =
          floatbuffer[i] * Scales[0] + Means[0];  // R
      rgbbuffer[i * 3 + idxRGB[1]] =
          floatbuffer[i + imgsize] * Scales[1] + Means[1];  // G
      rgbbuffer[i * 3 + idxRGB[2]] =
          floatbuffer[i + imgsize * 2] * Scales[2] + Means[2];  // B
    }
  }

  return size;
}

// Read ppm file and convert to float, copy data if batchsize > 1
int readPPMFile(const char *filename, unsigned char *&rgbbuffer, int &im_w,
                int &im_h, char *&databuffer, int modelType = 1) {
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

  float *floatbuffer = new float[size * batchsize];
  rgbbuffer = new unsigned char[size];
  // infile.seekg(1, infile.cur);
  infile.read((char *)rgbbuffer, 1);
  printf("Read P%c w=%s, h=%s, size=%dx%d m=%d magic=%02X\n", magic[1], w, h,
         size, batchsize, modelType, rgbbuffer[0]);

  infile.read((char *)rgbbuffer, size);

  for (int i = 0; i < imgsize; i++) {
    floatbuffer[i] =
        (rgbbuffer[i * 3 + idxRGB[0]] - Means[0]) / Scales[0];  // R
    floatbuffer[i + imgsize] =
        (rgbbuffer[i * 3 + idxRGB[1]] - Means[1]) / Scales[1];  // G
    floatbuffer[i + imgsize * 2] =
        (rgbbuffer[i * 3 + idxRGB[2]] - Means[2]) / Scales[2];  // B
  }
  for (int nb = 1; nb < batchsize; nb++) {
    memcpy(floatbuffer + nb * size, floatbuffer, size * sizeof(float));
  }

  databuffer = (char *)floatbuffer;
  // delete[] rgbbuffer;

  return 0;
}

int readPPMFile2(const char *filename, unsigned char *&rgbbuffer, int &im_w,
                 int &im_h, char *&databuffer, int modelType = 1) {
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
  im_h = atoi(h) / batchsize;
  im_w = atoi(w);

  int idxRGB[3] = {0, 1, 2};
  float Scales[3] = {1.f, 1.f, 1.f};
  float Means[3] = {0.f, 0.f, 0.f};
  getPreprocesParams(modelType, im_w, im_h, idxRGB, Scales, Means);

  int imgsize = im_w * im_h;
  int size = imgsize;
  if (magic[1] == '6') size *= 3;

  float *floatbuffer = new float[size * batchsize];
  rgbbuffer = new unsigned char[size * batchsize];
  // infile.seekg(1, infile.cur);
  infile.read((char *)rgbbuffer, 1);
  printf("Read P%c w=%s, h=%s, size=%dx%d m=%d magic=%02X\n", magic[1], w, h,
         size, batchsize, modelType, rgbbuffer[0]);

  infile.read((char *)rgbbuffer, size * batchsize);

  for (int b = 0; b < batchsize; b++) {
    int offset = b * imgsize * 3;
    for (int i = 0; i < imgsize; i++) {
      floatbuffer[i + offset] =
          (rgbbuffer[offset + i * 3 + idxRGB[0]] - Means[0]) / Scales[0];  // R
      floatbuffer[i + offset + imgsize] =
          (rgbbuffer[offset + i * 3 + idxRGB[1]] - Means[1]) / Scales[1];  // G
      floatbuffer[i + offset + imgsize * 2] =
          (rgbbuffer[offset + i * 3 + idxRGB[2]] - Means[2]) / Scales[2];  // B
    }
  }

  databuffer = (char *)floatbuffer;
  // delete[] rgbbuffer;

  return 0;
}

#ifndef USE_CV
int readImageByOpenCV(const char *filename, unsigned char *&rgbbuffer,
                      int &im_w, int &im_h, char *&databuffer,
                      int modelType = 1) {
  printf("readImageByOpenCV %s Empty Function\n", filename);
  return -1;
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
  float *floatbuffer = new float[imgsize * 3 * batchsize];
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

  int size = im_w * im_h * 3;
  for (int nb = 1; nb < batchsize; nb++) {
    memcpy(floatbuffer + nb * size, floatbuffer, size * sizeof(float));
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
  float *pMask;      // raw data of mask
  float *pKeypoint;  // raw data of keypoint

  // add by dyg
  float occupyScore;
  float leverScore;
  float lockScore;
  float *pLeverKp;
  float *pLockKp;
} TRoiMaskOut;
static int num_classes = 9;  // cfg.MODEL.NUM_CLASSES = 6
const int MaskSize = 28;
const int DETECTNUM = 100;
static int MaskBlockSize = 0;
static int park_kp_num = 10;  // number of keypoint, 8 or 10
static int lever_kp_num = 0;
static int lock_kp_num = 0;
static int box_prob_num = 5;  // box + cls

void ParseMaskData(float *pOutData, std::vector<TRoiMaskOut> &roiMasks) {
  roiMasks.clear();
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
  int kpNum = 10;  // total keypoint num
  val = getenv("TRT_KPNUM");
  if (NULL != val) {
    kpNum = atoi(val);
    printf("getenv TRT_KPNUM=%d\n", kpNum);
  }

  if (kpNum > 10) {
    box_prob_num = 8;  // box + cls + occupy/lever/lock
    park_kp_num = 8;
    lever_kp_num = 2;
    lock_kp_num = 2;
  } else {
    park_kp_num = kpNum;
  }

  const int MaskOutSize[5] = {
      postNMS * 5, postNMS * num_classes * 4,
      postNMS * num_classes * (box_prob_num - 4),
      DETECTNUM * (box_prob_num + 4 * 5 + 1),
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
        if (tmpData[box_prob_num * i] > 1e-5) {
          int classID =
              floor(tmpData[box_prob_num * i]);  // classID + 0.1f*roiScore
          float roiScore = (tmpData[box_prob_num * i] - classID) * 10.0f;
          float left = tmpData[box_prob_num * i + 1];
          float top = tmpData[box_prob_num * i + 2];
          float right = tmpData[box_prob_num * i + 3];
          float bottom = tmpData[box_prob_num * i + 4];
          // add by dyg
          float occupy = 0, lever = 0, lock = 0;
          if (box_prob_num == 8) {
            occupy = tmpData[box_prob_num * i + 5];
            lever = tmpData[box_prob_num * i + 6];
            lock = tmpData[box_prob_num * i + 7];
            printf("occupy=%f lever=%f lock=%f\n", occupy, lever, lock);
          }
          printf(
              "%d classID=%d roiScore=%f left=%f top=%f right=%f bottom=%f\n",
              i, classID, roiScore, left, top, right, bottom);
          roiNum_nms = i + 1;

          TRoiMaskOut tmpRoi = {classID,      roiScore,     left, top,
                                right - left, bottom - top, NULL, NULL,
                                occupy,       lever,        lock};
          roiMasks.push_back(tmpRoi);
        } else {
          break;
        }
      }
    }

    if (4 == outid) {                                          // masks
      const int maskSize = MaskSize * MaskSize * num_classes;  // float(4byte)
      for (int i = 0; i < roiNum_nms; i++) {
        // mask contours used Mask block 0
        roiMasks[i].pMask = tmpData;
        // keypoints used Mask block 1, each has 'kpNum' index&conf;
        // roiMasks[i].pKeypoint = tmpData + maskSize + i * kpNum * 2;
        roiMasks[i].pKeypoint = tmpData + maskSize + i * park_kp_num * 2;

        // add by dyg
        if (box_prob_num == 8) {
          roiMasks[i].pLeverKp = tmpData + maskSize +
                                 roiNum_nms * park_kp_num * 2 +
                                 i * lever_kp_num * 2;
          roiMasks[i].pLockKp =
              tmpData + maskSize + roiNum_nms * park_kp_num * 2 +
              roiNum_nms * lever_kp_num * 2 + i * lock_kp_num * 2;
        }
      }
    }

    tmpData += MaskOutSize[outid];
  }
}

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
      if (iy < 0 || iy >= im_h || x0 < 0 || x0 >= im_w) continue;
      int offset = (iy * im_w + x0) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = color[ch];
      }
    }
  } else {
    float scale = (float)(y1 - y0) / (x1 - x0);
    float dx = (fabs(scale) > 1.f) ? fabs(1 / scale) : 1.f;

    if (x0 > x1) {
      int tmp = x1;
      x1 = x0;
      x0 = tmp;
      y0 = y1;
    }

    for (float ix = x0; ix < x1; ix += dx) {
      int iy = (ix - x0) * scale + y0;
      if (ix < 0 || ix >= im_w || iy < 0 || iy >= im_h) continue;
      int offset = (iy * im_w + (int)ix) * 3;
      for (int ch = 0; ch < 3; ch++) {
        rgbbuffer[offset + ch] = color[ch];
      }
    }
  }
}

static int img_width = 1280;
static int img_height = 960;
static int keypoint_model_w = 640;
static int keypoint_model_h = 256;
static int perception_w = 640;
static int perception_h = 480;
static int roi_w = perception_w;
static int roi_h = 256;
static int roi_x = 0;
static int roi_y = perception_h - roi_h;

static const char *labels[5] = {"park_free-horizontal", "park_free-vertical",
                                "park_occupy-horizontal",
                                "park_occupy-vertical", "free_space"};

void Rescale(int &x, int &y) {
  x *= (float)roi_w / keypoint_model_w;
  y *= (float)roi_h / keypoint_model_h;
  x += roi_x;
  y += roi_y;
  x *= (float)img_width / perception_w;
  y *= (float)img_height / perception_h;
}
void Print2Json(int classID, std::vector<int> &vx, std::vector<int> &vy,
                std::vector<float> &vconf, std::ostream &fout = std::cerr) {
  // fprintf( fout, "{\"label\": \"%s\", \"points\": [[%d, %d, %.2f]\n");
  fout << "{\"label\": \"" << labels[classID - 1] << "\", \"points\": [";
  for (unsigned int i = 0; i < vx.size(); ++i) {
    if (i > 0) fout << ", ";
    if (vx[i] <= -100 && vconf[i] < -100.f) {  // empty value in freespace
      fout << "[]";
    } else {
      Rescale(vx[i], vy[i]);
      fout << "[" << vx[i] << ", " << vy[i] << ", " << vconf[i] << "]";
    }
  }
  fout << "], \"shape_type\": \"polygon\"}";
}

void DrawMaskResult(std::vector<TRoiMaskOut> &roiMasks,
                    unsigned char *rgbbuffer, int im_w, int im_h,
                    std::string &outfile) {
  const int im_size = im_w * im_h;
  const float im_scale = 960.0f / 960.0f;
  const int classColor[12][3] = {{0, 255, 0},   {31, 223, 0},  {255, 0, 0},
                                 {223, 31, 0},  {255, 224, 0}, {63, 63, 0},
                                 {255, 127, 0}, {127, 0, 64},  {0, 255, 127},
                                 {127, 255, 0}, {127, 0, 255}, {255, 255, 255}};

  std::ostream &jsonOut = std::cerr;
  std::ofstream jsonfile;
  std::streambuf *cerrbuf = jsonOut.rdbuf();
  if (!outfile.empty() && testInMemory) {  // save data to python code
    jsonfile.open(outfile + ".json", std::ios::out);
    jsonOut.rdbuf(jsonfile.rdbuf());
  }
  jsonOut.precision(6);
  jsonOut << "{\"shapes\": [";

  for (unsigned int i = 0; i < roiMasks.size(); i++) {
    float ox = roiMasks[i].x * im_scale;
    float oy = roiMasks[i].y * im_scale;
    float w = roiMasks[i].width * im_scale;
    float h = roiMasks[i].height * im_scale;
    // float *pMask = roiMasks[i].pMask;
    int classID = roiMasks[i].classID;
    int colorID = classID - 1;
    printf("Draw classID=%d  rect={%f,%f,%f,%f}\n", classID, ox, oy, w, h);
    {
      static float min_w = 32.0f, min_h = 32.0f;
      if (w < min_w * 1.2f || h < min_h * 1.2f) {
        printf("Find small roi=(%f, %f) min=(%f, %f)\n", w, h, min_w, min_h);
        min_w = std::min(w, min_w);
        min_h = std::min(h, min_h);
      }
    }

    float scale_x = w / MaskSize;
    float scale_y = h / MaskSize;
    int boarder = 2;
    int mask_width = (MaskSize + 2 * boarder) * 2;  // add boader and resize 2x
    int tx = -1, ty = -1;                           // pre point
    int tx0 = 0, ty0 = 0;                           // first point
    // Draw Mask
    if (classID == 5 || (box_prob_num == 8 && i == 0)) {
      std::vector<int> vx, vy;
      std::vector<float> vconf;
      float *pContour = roiMasks[0].pMask;
      for (int j = 0; j < (MaskSize * MaskSize); j++) {
        int val = (int)pContour[j];
        if (0 == val || -10 == val) break;

        // add by dyg
        if (box_prob_num == 8) {
          ox = 0;
          oy = 0;
          scale_x = im_w / 80.0;
          scale_y = im_h / 32.0;
          mask_width = (80 + 2 * boarder) * 2;
        }

        if (val != -1) {
          int iy = (val / mask_width + 0.5f) * scale_y / 2 + oy - 0.5f;
          int ix = (val % mask_width + 0.5f) * scale_x / 2 + ox - 0.5f;
          iy = std::min(im_h - 1, std::max(0, iy));
          ix = std::min(im_w - 1, std::max(0, ix));
          vx.push_back(ix);
          vy.push_back(iy);
          vconf.push_back(2.f);

          if (tx >= 0) {
            printf(" ,%d(%d,%d)", val, iy, ix);
            DrawLine(rgbbuffer, im_w, im_h, classColor[colorID], tx, ty, ix,
                     iy);
          } else {
            printf("%d Contour idx(y,x): %d(%d,%d)", i, val, iy, ix);
            tx0 = ix;
            ty0 = iy;
          }
          tx = ix;
          ty = iy;
        } else {
          vx.push_back(-100);
          vy.push_back(-100);
          vconf.push_back(-100.1f);

          DrawLine(rgbbuffer, im_w, im_h, classColor[colorID], tx, ty, tx0,
                   ty0);
          tx = -1;
          ty = -1;
        }
      }
      printf(" ,EndContour\n");
      if (0 < i) jsonOut << ", ";
      Print2Json(classID, vx, vy, vconf, jsonOut);
      if (classID == 5) continue;
    }

    // Draw Keypoint
    if (classID != 5) {
      int sx = -1000, sy = -1000;
      std::vector<int> vx, vy;
      std::vector<float> vconf;
      // float *pKP = roiMasks[i].pKeypoint;  // 10x index&conf, index= y*1e4 +x
      std::vector<int> kp_classes = {park_kp_num, lever_kp_num, lock_kp_num};
      std::vector<float *> pKPs = {roiMasks[i].pKeypoint, roiMasks[i].pLeverKp,
                                   roiMasks[i].pLockKp};
      for (unsigned int n = 0; n < kp_classes.size(); n++) {
        int kp_num = kp_classes[n];
        float *pKP = pKPs[n];

        int color[3] = {0, 0, 0};
        if (n == 0) {  // park
          if (roiMasks[i].occupyScore > 0.5) {
            color[0] = 255;
          } else {
            color[2] = 255;
          }
        }
        if (n == 1) {  // lever
          if (roiMasks[i].leverScore > 0.5) {
            color[0] = 128;
          } else {
            continue;
          }
        }
        if (n == 2) {  // lock
          if (roiMasks[i].lockScore > 0.5) {
            color[0] = 255;
            color[2] = 255;
          } else {
            continue;
          }
        }

        for (int j = 0; j < kp_num; j++) {
          float conf = pKP[(j % kp_num) * 2 + 1];
          short *pIndex = (short *)(pKP + (j % kp_num) * 2);
          int ix = pIndex[0];
          int iy = pIndex[1];
          printf("%d maxIdx [%d,%d] conf=%f\n", i, iy, ix, conf);
          vx.push_back(ix);
          vy.push_back(iy);
          vconf.push_back(conf);

          // iy = std::min(im_h - 1, std::max(0, iy));
          // ix = std::min(im_w - 1, std::max(0, ix));

          if (j > 0 && j < kp_num + 1) {
            if (n == 0 && roiMasks[i].occupyScore < 0.5)
              memcpy(color, classColor[j], 3 * sizeof(int));
            DrawLine(rgbbuffer, im_w, im_h, color, tx, ty, ix, iy);
          }
          tx = ix;
          ty = iy;
          if (-1000 == sx && -1000 == sy) {  // store the fist vaild point
            sx = ix;
            sy = iy;
          }
        }
      }
      if (0 < i) jsonOut << ", ";
      Print2Json(classID, vx, vy, vconf, jsonOut);

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

  jsonOut << "], \"version\": \"3.16.2.keypoint_annotation_0138\"}\n";
  if (jsonfile.is_open()) {
    jsonfile.flush();
    jsonfile.close();
    jsonOut.rdbuf(cerrbuf);
  }

  if (!testInMemory && !outfile.empty() && NULL != rgbbuffer) {
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
  }
}
// process the output of default model-----------------------------------------
void DrawDefaultResult(float *LLDoutput, unsigned char *rgbbuffer, int im_w,
                       int im_h, const char *outfile) {
  const int im_size = im_w * im_h;

  if (!testInMemory && NULL != outfile && NULL != rgbbuffer) {  // draw in image
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
  }

  unsigned int valNum = outputSize / sizeof(float);
  if (valNum > 0 && NULL != outfile) {
    std::string filename = outfile;
    cnpy::npy_save(filename + ".npy", LLDoutput, {valNum});
  }
}

// process code for xp_model.onnx ---------------------------------------------
struct Config {
  // LLD
  int num_points_each_line = 32;  // you need to define it
  float scale = 2.0;              // scale of label
  int start_y = 470;
  int end_y = 210;                           // ROI(rom buttom to top) 344
  int number_of_lanes = 7;                   // currently support up to 5 lanes
  int num_bits_each_lane = 104;              // 100
  int binary_end = 8;                        // 6
  int left_line_type_end = binary_end + 14;  //
  int right_line_type_end = left_line_type_end + 14;
  int left_points_end = right_line_type_end + 2 + num_points_each_line;
  int right_points_end = left_points_end + 2 + num_points_each_line;
  unsigned int num_bits_all_lanes = 729;
} config;

int visulizeLaneNet(unsigned char *image, int iw, int ih, float *label,
                    const float *mask, int modelType = 4) {
  // int point_size = 3;
  int colors[5][3] = {
      {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {255, 0, 255}, {255, 255, 0},
  };
  int delta_y = (config.start_y - config.end_y) / config.num_points_each_line /
                config.scale;

  int left_start = config.right_line_type_end + 2;
  int right_start = left_start + config.num_points_each_line + 2;
  int range_offset = iw / 2;
  for (int j = 0; j < config.number_of_lanes; j++) {
    float lane_exists = label[0 + j * config.num_bits_each_lane];
    if (lane_exists < 1e-5) continue;  // check if lane exists

    int start_offset[2] = {left_start, right_start};
    for (int s = 0; s < 2; s++) {
      int start = start_offset[s] + j * config.num_bits_each_lane;
      label[start - 1] = label[start - 1] + range_offset;
      label[start - 2] = label[start - 2] + range_offset;
      std::cout << "sx-top:" << label[start - 1]
                << " sx-bottom:" << label[start - 2] << std::endl;
      for (int i = 0; i < config.num_points_each_line; i++) {
        // if(mask[start + i] > 0)
        {
          float px = label[start + i] + range_offset;
          float py = config.start_y / config.scale - delta_y * i;
          // check the range
          if (px < 0 || px >= iw || py < label[start - 2] ||
              py > label[start - 1])
            continue;

          if (i < 2 || i > config.num_points_each_line - 2)
            std::cout << "px:" << px << " py:" << py << std::endl;
          // cv2.circle(image, p, point_size, colors[j], -1)
          if (NULL != image) {
            int offset = ((int)py * iw + (int)px) * 3;
            for (int ch = 0; ch < 3; ch++) {
              int colorval = colors[j][2 - ch];
              image[offset + ch] = colorval;
              if (py < ih - 1 && px < iw - 1) {  // check the border
                image[offset + ch + 3] = colorval;
                image[offset + ch + iw * 3] = colorval;
                image[offset + ch + iw * 3 + 3] = colorval;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}

void DrawLaneNetResult(float *LLDoutput, unsigned char *rgbbuffer, int im_w,
                       int im_h, const char *outfile, int modelType = 4) {
  visulizeLaneNet(rgbbuffer, im_w, im_h, LLDoutput, NULL, modelType);
  const int im_size = im_w * im_h;
  if (!testInMemory && NULL != outfile && NULL != rgbbuffer) {  // draw in image
    std::ofstream file(outfile, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
  }

  if (NULL != outfile) {  // save data to python code
    std::string filename = outfile;
    cnpy::npy_save(filename + ".npy", LLDoutput, {config.num_bits_all_lanes});
  }
}

// process the output of MOD----------------------------------------
#include <algorithm>
#include <memory>
using namespace std;

// ported from
// https://gitlab.xiaopeng.us/xpilot/xpilot_vision/blob/master/utf2/configs/release/config_20190517.yaml
struct MODConfig {
  // int image_width = 457;
  // int image_height = 237;

  // Architecture
  int grid_width = 29;
  int grid_height = 15;

  // Post-processing
  float detection_threshold = 0.73f;
  float overlap_threshold = 0.9f;
  float iou_threshold = 0.5f;  // For nms
  int max_num_box = 20;        // nms
  int min_box_area_for_kpi = 225;

  // Anchor Generation
  int boxes_per_cell = 4;
  int num_classes = 12;
  int num_attr = 0;
  int num_boxes = 0;
  float anchors_wh[7][2] = {{20., 50.},   {40., 80.}, {80, 40},  {60., 100.},
                            {250., 300.}, {250, 200}, {200, 150}};

} MODCfg;

struct Rectf {
  float x;
  float y;
  float width;
  float height;
};

int filterOverlappingNMS(Rectf *dets, float *scores, std::vector<int> idx,
                         float thresh = 0.5f, const int nms_topN = 20,
                         const int min_box_area = 225) {
  int boxnum = idx.size();
  std::vector<float> areas(boxnum);
  for (int ti = 0; ti < boxnum; ti++) {
    int i = idx[ti];
    areas[i] = (float)dets[i].width * dets[i].height;
  }

  for (int ti = 0; ti < boxnum; ti++) {
    int i = idx[ti];
    if (scores[i] <= 0.0f) continue;
    for (int tj = 0; tj < ti; tj++) {
      int j = idx[tj];
      if (scores[j] <= 0.0f) continue;
      float xx1 = std::max(dets[j].x, dets[i].x);
      float yy1 = std::max(dets[j].y, dets[i].y);
      float xx2 =
          std::min(dets[j].x + dets[j].width, dets[i].x + dets[i].width);
      float yy2 =
          std::min(dets[j].y + dets[j].height, dets[i].y + dets[i].height);
      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float inter = w * h;
      if (inter > MODCfg.overlap_threshold * areas[i]) {
        // std::cout << "rm overlopping i: "<< scores[i] << ", inter=" <<
        // inter/areas[i]  << std::endl;
        scores[i] = -scores[i];
      } else if (inter > MODCfg.overlap_threshold * areas[j]) {
        // std::cout << "rm overloping j: "<< scores[j] << ", inter=" <<
        // inter/areas[j] << std::endl;
        scores[j] = -scores[j];
      }
    }
  }

  int roiIdx = 0;  // roi index post nms
  for (int ti = 0; ti < boxnum; ti++) {
    int i = idx[ti];
    if (roiIdx >= nms_topN)
      scores[i] = -10.0f - scores[i];  // keep no more then nms_topN boxes
    if (scores[i] <= 0.0f) continue;

    for (int tj = 0; tj < ti; tj++) {
      int j = idx[tj];
      if (scores[j] <= 0.0f) continue;
      float xx1 = std::max(dets[j].x, dets[i].x);
      float yy1 = std::max(dets[j].y, dets[i].y);
      float xx2 =
          std::min(dets[j].x + dets[j].width, dets[i].x + dets[i].width);
      float yy2 =
          std::min(dets[j].y + dets[j].height, dets[i].y + dets[i].height);
      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float inter = w * h;

      float iouV = inter / (areas[j] + areas[i] - inter);
      if (iouV > thresh) {
        scores[i] = -scores[i];  // drop boxes have larger iouV then thresh;
        // std::cout << "rm nms: "<< i << " vs " << j<<" , iouV=" <<iouV <<
        // std::endl;
        break;
      }
    }

    if (scores[i] > 0.0f) {
      roiIdx++;
    }
  }

  for (int ti = 0; ti < boxnum; ti++) {
    int i = idx[ti];
    if (scores[i] > 0.0f && areas[i] < min_box_area)
      scores[i] /= 10.0f;  // drop small boxes , confidence < 0.1
  }

  return 0;
}

void genAnchorBoxes(int image_width, int image_hight, int grid_width,
                    int grid_hight, Rectf *anchors) {
  float cell_size_x = float(image_width) / grid_width;
  float cell_size_y = float(image_hight) / grid_hight;
  if (10 == grid_hight) {
    for (int i = 0; i < grid_width; i++) {
      for (int j = 0; j < grid_hight; j++) {
        float x = i * cell_size_x;
        float y = j * cell_size_y;
        for (int b = 0; b < MODCfg.boxes_per_cell; b++) {
          anchors->x = x;
          anchors->y = y;
          anchors->width = MODCfg.anchors_wh[b][0];
          anchors->height = MODCfg.anchors_wh[b][1];
          anchors++;
        }
      }
    }
  } else {
    for (int i = 0; i < grid_hight; i++) {
      float y = i * cell_size_y;
      for (int j = 0; j < grid_width; j++) {
        float x = j * cell_size_x;
        for (int b = 0; b < MODCfg.boxes_per_cell; b++) {
          anchors->x = x;
          anchors->y = y;
          anchors->width = MODCfg.anchors_wh[b][0];
          anchors->height = MODCfg.anchors_wh[b][1];
          anchors++;
        }
      }
    }
  }
}

inline float sigmoid(float x) { return (1 / (1 + expf(-x))); }
inline float clamp(float x) { return max(0.f, min(1.f, x)); }

Rectf getBoxCorners(int iw, int ih, int grid_width, int grid_hight,
                    Rectf *anchor_boxes, int box_id, float *bbox) {
  float x = sigmoid(bbox[0]);
  float y = sigmoid(bbox[1]);
  float w = bbox[2];
  float h = bbox[3];
  float cell_size_x = float(iw) / grid_width;
  float cell_size_y = float(ih) / grid_hight;

  float x_cell = anchor_boxes[box_id].x;
  float y_cell = anchor_boxes[box_id].y;

  float wa = anchor_boxes[box_id].width;
  float ha = anchor_boxes[box_id].height;

  float x_p = x * cell_size_x + x_cell;
  float y_p = y * cell_size_y + y_cell;

  float w_p =
      expf(min(w, 5.0f)) * wa;  // diff1: min(expf(min(w, 20.0f))*wa, 1000.0f);
  float h_p =
      expf(min(h, 5.0f)) * ha;  // diff1: min(expf(min(h, 20.0f))*ha, 1000.0f);

  float x_min = max(0.0f, x_p - w_p / 2.f);  // diff2:  int(w_p/2.f)
  float y_min = max(0.0f, y_p - h_p / 2.f);  // diff2:  int(h_p/2.f)

  float x_max = min(max(0.0f, x_p + w_p / 2.f), float(iw - 1));
  float y_max = min(max(0.0f, y_p + h_p / 2.f), float(ih - 1));

  Rectf rect;
  rect.x = int(x_min);
  rect.y = int(y_min);
  rect.width = int(x_max) - rect.x;
  rect.height = int(y_max) - rect.y;

  return rect;
}

static Rectf *g_anchorBuffer = nullptr;
int visulizeMOD(unsigned char *image, int iw, int ih, float *output) {
  const float conf_th = MODCfg.detection_threshold;
  const int offset = (5 + MODCfg.num_classes + MODCfg.num_attr);
  int grid_width = MODCfg.grid_width;
  int grid_hight = MODCfg.grid_height;
  int anchorNum = grid_width * grid_hight * MODCfg.boxes_per_cell;
  g_anchorBuffer = new Rectf[anchorNum];
  genAnchorBoxes(iw, ih, grid_width, grid_hight, g_anchorBuffer);

  auto t_start = std::chrono::high_resolution_clock::now();

  std::shared_ptr<Rectf> boxesBuffer(new Rectf[anchorNum]);
  Rectf *boxes = boxesBuffer.get();
  std::shared_ptr<float> confsBuffer(new float[anchorNum]);
  float *confs = confsBuffer.get();
  int boxnum = 0;
  for (int box_id = 0; box_id < anchorNum; box_id++) {
    float conf =
        sigmoid(output[box_id * offset + 4]);  // sigmoid for xp_model_v0.3
    if (conf > conf_th) {
      boxes[boxnum] =
          getBoxCorners(iw, ih, grid_width, grid_hight, g_anchorBuffer, box_id,
                        &output[box_id * offset]);
      confs[boxnum] = conf;
      // std::cout << boxnum << "," << confs[boxnum] << "," << boxes[boxnum].x
      // << "," << boxes[boxnum].y
      //        << "," << boxes[boxnum].x + boxes[boxnum].width << ","
      //        << boxes[boxnum].y + boxes[boxnum].height << std::endl;
      boxnum++;
    }
  }

  // NMS
  std::vector<int> idx(boxnum);
  for (unsigned int i = 0; i != idx.size(); ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(),
                   [&confs](int i1, int i2) { return confs[i1] > confs[i2]; });
  filterOverlappingNMS(boxes, confs, idx, MODCfg.iou_threshold,
                       MODCfg.max_num_box, MODCfg.min_box_area_for_kpi);
  // NonMaxSuppression(boxScores, 0.5f, conf_th);

  auto t_end = std::chrono::high_resolution_clock::now();
  float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  printf("mynms time = %f ms\n", ms);

  int colors[2][3] = {{0, 255, 0}, {255, 255, 255}};
  for (int ti = 0; ti < boxnum; ti++) {
    int i = idx[ti];
    if (confs[i] <= 0.0f) continue;
    Rectf &objABox = boxes[i];
    std::cout << i << "," << confs[i] << "," << objABox.x << "," << objABox.y
              << "," << objABox.x + objABox.width << ","
              << objABox.y + objABox.height << std::endl;
    int col = 0;
    if (confs[i] < conf_th) col = 1;  // small boxes, confidence < 0.1

    // Draw box
    if (NULL == image) continue;

    for (int x = 0; x < objABox.width; x++) {
      int offset = (int(objABox.y) * iw + int(objABox.x + x)) * 3;
      for (int ch = 0; ch < 3; ch++) {
        image[offset + ch] = colors[col][ch];
        image[offset + int(objABox.height) * iw * 3 + ch] = colors[col][ch];
      }
    }
    for (int y = 0; y < objABox.height; y++) {
      int offset = (int(objABox.y + y) * iw + int(objABox.x)) * 3;
      for (int ch = 0; ch < 3; ch++) {
        image[offset + ch] = colors[col][ch];
        image[offset + int(objABox.width) * 3 + ch] = colors[col][ch];
      }
    }
  }
  delete[] g_anchorBuffer;

  return 0;
}

void DrawMODNetResult(float *MODoutput, unsigned char *rgbbuffer, int im_w,
                      int im_h, const char *outfile) {
  std::cout << "DrawMODNetResult" << std::endl;
  visulizeMOD(rgbbuffer, im_w, im_h, MODoutput);

  std::string filename = outfile;
  const int im_size = im_w * im_h;
  if (!testInMemory && NULL != outfile && NULL != rgbbuffer) {  // draw in image
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P6\n" << im_w << " " << im_h << "\n255\n";
    file.write((char *)rgbbuffer, im_size * 3);
    file.close();
    std::cout << "Save PPM: " << filename << std::endl;
  }

  if (NULL != outfile) {  // save data to csv file
    unsigned int numboxes = MODCfg.num_boxes;
    unsigned int offset = (5 + MODCfg.num_classes + MODCfg.num_attr);
    cnpy::npy_save(filename + ".npy", MODoutput, {numboxes, offset});
  }

  return;
}

// print the fist & last 32(PrintNum) data
void printOutput(int64_t eltCount, float *outputs) {
  const int PrintNum = 32;
  for (int64_t eltIdx = 0; eltIdx < eltCount && eltIdx < PrintNum; ++eltIdx) {
    std::cerr << outputs[eltIdx] << "\t, ";
  }
  if (eltCount > PrintNum) {
    std::cerr << " ... ";
    for (int64_t eltIdx = (PrintNum - PrintNum); eltIdx < eltCount; ++eltIdx) {
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

float percentile(float percentage, std::vector<float> &times) {
  int all = static_cast<int>(times.size());
  int exclude = static_cast<int>((1 - percentage / 100) * all);
  if (all > 10 && 0 <= exclude && exclude <= all) {
    std::sort(times.begin(), times.end());
    float pctTime = times[all == exclude ? 0 : all - 1 - exclude];
    float totTime = 0;
    for (int i = 5; i < all - 5; i++)
      totTime += times[i];  // drop the fist & last 5 datas.

    printf(
        "TestAll %d range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< "
        "%.3fms\n",
        all, times[5], times[all - 6], totTime / (all - 10), times[all / 2],
        percentage, pctTime);

    fprintf(
        stderr,
        "TestAll %d range=[%.3f, %.3f]ms, avg=%.3fms, 50%%< %.3fms, %.0f%%< "
        "%.3fms\n",
        all, times[5], times[all - 6], totTime / (all - 10), times[all / 2],
        percentage, pctTime);

    return pctTime;
  }
  return std::numeric_limits<float>::infinity();
}

int main(int argc, char *argv[]) {
  printf(
      "usage: test [model.trt] [./libonnxtrt.so] [inputFloat32.dat/image.ppm] "
      "[ LLD | MaskNetWeight.bin | LLDDS | SOFTISP | KPI] "
      "[output.ppm] \n");

  const char *pSoName = "./libonnxtrt.so";
  const char *trtmodel = "model_320x576b2.trt";
  const char *pMaskWeight = NULL;  //"fastrcnn_no_weight";

  if (argc > 1) trtmodel = argv[1];
  if (argc > 2) pSoName = argv[2];
  if (argc > 4) pMaskWeight = argv[4];

  int outType = 0;    // 0: plan 2 : packed by batchsize
  int modelType = 0;  // 0: LLD, 1: MaskRCNN/FasterRCNN, 2: LaneNet, 3 MODNet, ,
                      // 4 Lane & MOD
  // Mask or mask mean MaskRCNN, if can't read MaskWeight
  // the only get box ( like FasterRCNN )
  // if (0 == strncasecmp(pMaskWeight, "Mask", 4)) {
  if (NULL != strstr(pMaskWeight, "mask") ||
      NULL != strstr(pMaskWeight, "kpnet") ||
      NULL != strstr(pMaskWeight, "parking") ||
      NULL != strstr(pMaskWeight, "fsd")) {
    modelType = 1;
    outType = 1;
  } else if (0 == strncasecmp(pMaskWeight, "LaneNet", 7)) {
    modelType = 2;
  } else if (0 == strncasecmp(pMaskWeight, "MODNet", 6)) {
    modelType = 3;
  } else if (0 == strncasecmp(pMaskWeight, "LaneMOD", 7)) {
    modelType = 4;
    MODCfg.grid_width = 15;
    MODCfg.grid_height = 10;
    MODCfg.num_classes = 0;
    MODCfg.detection_threshold = 0.9f;
  } else if (0 == strncasecmp(pMaskWeight, "LLDMOD", 6)) {
    modelType = 5;
    MODCfg.grid_width = 30;
    MODCfg.grid_height = 19;
    MODCfg.num_classes = 0;
    MODCfg.detection_threshold = 0.9f;
  } else if (0 == strncasecmp(pMaskWeight, "LLDCLS12", 6)) {
    modelType = 6;
    MODCfg.num_classes = 12;
  } else if (0 == strncasecmp(pMaskWeight, "LLDDS", 5)) {
    modelType = 7;
    MODCfg.num_classes = 12;
    MODCfg.num_attr = 7;
    MODCfg.num_boxes = 4350;
    MODCfg.grid_width = 29;
    MODCfg.grid_height = 15;
  } else if (0 == strncasecmp(pMaskWeight, "SOFTISP", 7)) {
    modelType = 8;
  } else if (0 == strncasecmp(pMaskWeight, "KPI", 3)) {
    modelType = 9;
  }

  if (0 == MODCfg.num_boxes) {
    MODCfg.num_boxes =
        MODCfg.grid_width * MODCfg.grid_height * MODCfg.boxes_per_cell;
  }

  std::string outfile = "outimage.ppm";
  std::string inputfile = argv[3];
  int engineID = 0;  // ID for multi-engines
  int nImageNum = 0;
  std::vector<std::string> mImageList;
  if (argc > 5) {
    outfile = argv[5];
    if (0 == strncmp(argv[5], "/dev/shm", 8)) testInMemory = true;
  }

  int nMaxLoad = 1;
  {  // Set nMaxLoad >= 1, Test so load, initalize and release
    char *val = getenv("TRT_LOAD");
    if (NULL != val) nMaxLoad = atoi(val);
  }

  for (int nLoad = 0; nLoad < nMaxLoad; nLoad++) {
    void *pLibs = dlopen(pSoName, RTLD_LAZY);
    if (NULL == pLibs) {
      printf("Can not open library %s\n", pSoName);
      return -1;
    }

    pCreateEngine = (PTCreateEngine)dlsym(pLibs, "CreateEngine");
    pRunEngine = (PTRunEngine)dlsym(pLibs, "RunEngine");
    pDestoryEngine = (PTDestoryEngine)dlsym(pLibs, "DestoryEngine");
    pGetBuffer = (PTGetBufferOfEngine)dlsym(pLibs, "GetBufferOfEngine");
    if (argc > 3) {  // Read every line as image path from input file
      if (0 == inputfile.compare(inputfile.size() - 4, 4, ".txt")) {
        // Read every line everypath from input file
        std::ifstream infile(argv[3]);
        if (infile.good()) {
          std::string imgPath;
          while (std::getline(infile, imgPath)) {
            if (imgPath.length() > 2) mImageList.push_back(imgPath);
          }
        }
      } else {
        mImageList.push_back(inputfile);
      }
      nImageNum = mImageList.size();
    }

    char *val = getenv("TRT_BATCH");
    if (NULL != val) {
      batchsize = atoi(val);
      printf("getenv TRT_BATCH=%d\n", batchsize);
    }

    if (NULL != pCreateEngine && NULL != pRunEngine && NULL != pDestoryEngine) {
      int ret = pCreateEngine(engineID, trtmodel, pMaskWeight);
      printf("CreateEngine = %d\n", ret);
      std::vector<float> times;
      int bufferNum = 0;
      EngineBuffer *bufferInfo;
      if (NULL != pGetBuffer) {  // Get from GetBufferOfEngine
        const char *sBufferType[2] = {"In:", "Out:"};
        const char *sDataType[4] = {"FP32", "FP16", "INT8", "INT32"};
        pGetBuffer(engineID, &bufferInfo, &bufferNum, NULL);
        printf("GetBuffer num = %d\n", bufferNum);
        for (int i = 0; i < bufferNum; i++) {
          printf("Buf[%d] %s %dx[%d,%d,%d], %s\n", i,
                 sBufferType[bufferInfo[i].nBufferType], batchsize,
                 bufferInfo[i].d[0], bufferInfo[i].d[1], bufferInfo[i].d[2],
                 sDataType[bufferInfo[i].nDataType]);
          if (1 == bufferInfo[i].nBufferType) {       // Output buffer
            int outSize = sizeof(float) * batchsize;  // <= trt MaxBatch
            for (int j = 0; j < bufferInfo[i].nDims; j++)
              outSize *= bufferInfo[i].d[j];
            outputSize += std::min(bufferInfo[i].nBufferSize, outSize);
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

      float *pOutData = new float[outputSize / sizeof(float) * batchsize];
      for (int nT = max(1, nImageNum); nT > 0;
           nT--) {  // nT=1000, memory~=1.524GB
        int im_w = 0;
        int im_h = 0;
        unsigned char *rgbbuffer = NULL;
        char *inputStream = NULL;
        if (nImageNum > 0) {
          std::string input_filename = mImageList[(nT - 1) % nImageNum];
          if (!input_filename.empty()) {
            int namelen = input_filename.size() - 4;
            if (0 == input_filename.compare(namelen, 4, ".ppm")) {
              ret = readPPMFile(input_filename.c_str(), rgbbuffer, im_w, im_h,
                                inputStream, modelType);
              printf("readPPMFile %s ret=%d w=%d, h=%d, modelType=%d\n",
                     input_filename.c_str(), ret, im_w, im_h, modelType);
            } else if (0 == input_filename.compare(namelen, 4, ".bin") ||
                       0 == input_filename.compare(namelen, 4, ".raw")) {
              ret = ReadBinFile(input_filename.c_str(), rgbbuffer, im_w, im_h,
                                inputStream, modelType);
              printf("ReadBinFile %s size=%d Bytes w=%d h=%d\n",
                     input_filename.c_str(), ret, im_w, im_h);
            } else {  // Using Opencv to read other format
              ret = readImageByOpenCV(input_filename.c_str(), rgbbuffer, im_w,
                                      im_h, inputStream, modelType);
            }
            if (ret < 0) continue;
          }
          int nameLen = input_filename.size() - 4;
          input_filename =
              input_filename.substr(0, (nameLen > 0) ? nameLen : 0);
          if (argc <= 5)
            outfile = input_filename + "_out.ppm";
          else if (testInMemory) {
            outfile =
                input_filename.substr(input_filename.find_last_of("/") + 1);
            outfile = argv[5] + outfile + "_out";
          }
        }

        for (int i = 0; i < 3; i++) {
          auto t_start = std::chrono::high_resolution_clock::now();
          // Call tensorRT from C interface.
          ret = pRunEngine(engineID, batchsize, inputStream, 0,
                           (char *)pOutData, outType);

          float ms = std::chrono::duration<float, std::milli>(
                         std::chrono::high_resolution_clock::now() - t_start)
                         .count();
          printf("RunEngine ret = %d , time = %f ms\n", ret, ms);
          times.push_back(ms);
        }

        for (int nb = 0; nb < batchsize; nb++) {  // process for every batch
          int offsetBatch = outputSize * nb / sizeof(float);
          if (1 == modelType && 1 == outType) {
            std::vector<TRoiMaskOut> roiMasks;
            ParseMaskData(pOutData + nb * MaskBlockSize, roiMasks);
            if (NULL != rgbbuffer) {
              DrawMaskResult(roiMasks, rgbbuffer, im_w, im_h, outfile);
              // if (nb > 0) outfile.insert(outfile.find_last_of("."), "_" +
              // std::to_string(nb)); DrawMaskResult(roiMasks, rgbbuffer + nb *
              // im_w*im_h*3, im_w, im_h, outfile);
            }
          } else if (2 == modelType) {
            DrawLaneNetResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str());
          } else if (3 == modelType) {
            DrawMODNetResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str());
          } else if (4 == modelType) {
            DrawMODNetResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str());
            int numboxes =
                MODCfg.grid_width * MODCfg.grid_height * MODCfg.boxes_per_cell;
            int offsetMOD = numboxes * 5;  // input 480x302 -> output 19x30x4*5
            DrawLaneNetResult(pOutData + offsetMOD, rgbbuffer, im_w, im_h,
                              outfile.c_str());
          } else if (5 == modelType || 6 == modelType) {
            DrawLaneNetResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str(),
                              modelType);
            int offsetLLD =
                config.num_bits_all_lanes;  // input457x237->output 500
            DrawMODNetResult(pOutData + offsetLLD, rgbbuffer, im_w, im_h,
                             outfile.c_str());
          } else if (7 == modelType) {
            DrawLaneNetResult(pOutData + offsetBatch, rgbbuffer, im_w, im_h,
                              outfile.c_str(), modelType);
          } else {
            DrawDefaultResult(pOutData, rgbbuffer, im_w, im_h, outfile.c_str());
          }
        }
        if (!testInMemory)
          PrintOutBuffer(pOutData, outType, batchsize, bufferNum, bufferInfo);

        if (0 == (nT % 10))
          fprintf(stderr, "Inference %d/%d \r", nT, nImageNum);
        if (NULL != inputStream) delete[] inputStream;
        if (NULL != rgbbuffer) delete[] rgbbuffer;
      }
      delete[] pOutData;

      percentile(90, times);
      pDestoryEngine(engineID);
    }

    dlclose(pLibs);
  }

  return 0;
}
