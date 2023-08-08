////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//    Copyright© 2019 Xsense.ai Inc., Xmotors.ai Inc. and Xiaopeng Motors,    //
//                           All Rights Reserved.                             //
//                                                                            //
//  All users are hereby notified that the materials in the form of digital   //
//  information available from this software (content, designs, color         //
//  schemes, graphic styles, images, logo, text, and videos) comes protected  //
//  under International Copyright Laws. Therefore it should not be reproduced //
//  in any form digital or offline without prior written permission of        //
//  Xsense.ai Inc., Xmotors.ai Inc. and Xiaopeng Motors.                      //
//                                                                            //
//  Any unauthorized reprint or material usage (Xsense.ai Inc., Xmotors.ai    //
//  Inc. and Xiaopeng Motors) either manually or digitally, is strictly       //
//  prohibited.                                                               //
//                                                                            //
//  Any further unauthorized digital copying of this material via copying,    //
//  publication, reproduction or distribution of copyrighted works is an      //
//  infringement of the copyright owners' rights may be the subject of the    //
//  copyright of performers' protection under the Copyright Act. For such     //
//  illegal activities you will be strictly liable to Xsense.ai Inc.,         //
//  Xmotors.ai Inc. and Xiaopeng Motors for any and/or all damages (including //
//  recovery of attorneys' fees) which may be suffered and/ or incurred as a  //
//  result of your infringement.                                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gason.h"
#include "inference_json_generator.h"
#include "json_builder.h"
#include "json_formatter.h"
// #include "inference_decoder.hpp"

/*
  TODO (REZA):
    1. Add DDS reader to get result of inference
    2. Get ground truth data
    3. Generate ground truth and prediction json files
    4. Run KPI evaluation script
*/

using namespace xpilot::perception;

std::vector<float> GetInput(std::string path) {
  std::vector<float> input_array;
  std::ifstream input_stream(path.c_str());
  if (!input_stream.is_open()) {
    std::cerr << "Can not open input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  float input_element;
  while (input_stream >> input_element) {
    input_array.push_back(input_element);
  }
  input_stream.close();
  return input_array;
}

int main(int argc, char const *argv[]) {
  // std::vector<float> prediction_raw = GetInput("inference_output.txt");
  InferenceJsonGenerator generator("template_pred_mod.txt", "");
  generator.GenerateMODJson("");
  // float *labels;
  // std::vector<Box2D> boxes;
  // InferenceDecoder::Decode(prediction_raw.data(), labels, boxes);

  return 0;
}
