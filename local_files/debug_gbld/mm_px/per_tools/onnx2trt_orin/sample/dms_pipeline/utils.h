
#ifndef _UTILS_H_
#define _UTILS_H_
#include <cstring>  // memcpy
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
bool readBinToBuffer(const std::string fileName, char *pBuffer, int size) {
  uint32_t fileSize;
  std::ifstream is(fileName, std::ifstream::binary | std::ios::ate);
  if (pBuffer == nullptr) return false;
  fileSize = is.tellg();
  if (fileSize > size) return false;
  is.seekg(0, is.beg);
  is.read(pBuffer, fileSize);
  is.close();
  return true;
}
int32_t ReadBinFile(std::string filename, char *&databuffer) {
  int32_t size{0};
  std::ifstream file(filename, std::ios::binary);
  if (file.good() && filename.length() > 4) {
    file.seekg(0, file.end);
    size = file.tellg();
    databuffer = new char[size];
    file.seekg(0, file.beg);
    file.read(databuffer, size);
    file.close();
    if (nullptr == databuffer) {
      return -1;
    }
  }
  return size;
}
void getTestFiles(std::string list_path, std::vector<std::string> &file_list) {
  file_list.clear();
  std::ifstream list_file(list_path, std::ios::in);
  std::string temp;
  while (getline(list_file, temp)) {
    file_list.push_back(temp);
  }
  list_file.close();
}

void writeToFile(const char *input_buffer, int size, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  file.write(input_buffer, size);
  file.close();
}

void saveOutToFile(void *output_buffer[], int out_num, int out_size_sum, int *out_size, std::string output_path) {
  char *out = new char[out_size_sum];
  char *out_off = out;
  for (int i = 0; i < out_num; i++) {
    memcpy(out_off, output_buffer[i], out_size[i]);
    out_off += out_size[i];
  }
  writeToFile(out, out_size_sum, output_path);

  delete[] out;
}
void DrawBboxGray(uint8_t *databuffer, int im_w, int im_h, int xmin, int ymin, int xmax, int ymax) {
  xmin = xmin < 0 ? 0 : xmin;
  ymin = ymin < 0 ? 0 : ymin;
  xmax = xmax > (im_w - 1) ? (im_w - 1) : xmax;
  ymax = ymax > (im_h - 1) ? (im_h - 1) : ymax;
  int offset1 = ymin * im_w + xmin;
  int offset2 = ymax * im_w + xmin;

  for (int i = 0; i < xmax - xmin; i++) {
    databuffer[int(offset1) + i] = 255;
    databuffer[int(offset2) + i] = 255;
  }

  float offset3 = ymin * im_w + xmax;
  for (int j = 0; j < ymax - ymin; j++) {
    databuffer[int(offset1) + j * im_w] = 255;
    databuffer[int(offset3) + j * im_w] = 255;
  }
}
void DrawLandmarkGray(uint8_t *databuffer, float *landmarks, int landmark_num, int img_w, int img_h, int x_start,
                      int y_start, int crop_h, int crop_w) {
  for (int i = 0; i < landmark_num; i++) {
    int offset = i * 2;
    int x = landmarks[offset] * crop_w + x_start;
    int y = landmarks[offset + 1] * crop_h + y_start;
    if (1 < x && x < (img_w - 1) && 1 < y && y < (img_h - 1)) {
      // draw points
      int pixel_offset = y * img_w + x;
      databuffer[pixel_offset] = 255;

      databuffer[pixel_offset - 1] = 255;
      databuffer[pixel_offset + 1] = 255;
      databuffer[pixel_offset - img_w] = 255;
      databuffer[pixel_offset + img_w] = 255;
    }
  }
}
void writePPMGray(void *databuffer, int im_w, int im_h, const std::string &outfile) {
  std::ofstream file(outfile, std::ios::out | std::ios::binary);
  file << "P5\n" << im_w << " " << im_h << "\n255\n";
  file.write((char *)databuffer, im_w * im_h);
  file.close();
}

#endif  // end of _UTILS_H_
