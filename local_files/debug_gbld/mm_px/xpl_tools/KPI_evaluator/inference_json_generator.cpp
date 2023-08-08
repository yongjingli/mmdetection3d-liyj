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

#include "inference_json_generator.h"

namespace xpilot {
namespace perception {

InferenceJsonGenerator::InferenceJsonGenerator(std::string mod_template_path,
                                               std::string lld_template_path) {
  if (mod_template_path != "") LoadMODTemplate(mod_template_path);
  if (lld_template_path != "") LoadLLDTemplate(lld_template_path);
}

bool InferenceJsonGenerator::GenerateMODJson(std::string output_path) {
  // TODO (Reza): get result from inference decoder and put it in template
  // TODO (Reza): find a way to copy template. we don't want to change template.
  JsonValue test = mod_json_template_;
  begin(test)["objects"][0]["poly"][0]["x"]->value = JsonValue(1);
  std::cout << JsonFormatter::GetFormattedString(test) << std::endl;

  return false;
}

bool InferenceJsonGenerator::GenerateLLDJson(std::string output_path) {
  // TODO (Reza): get result from inference decoder and put it in template
  return false;
}

bool InferenceJsonGenerator::LoadMODTemplate(std::string template_path) {
  std::vector<char> buffer;
  if (!ReadFile(template_path, buffer)) {
    std::cerr << "Can not read template file" << std::endl;
    return false;
  }

  char *endptr;
  int parse_status = jsonParse(buffer.data(), &endptr, &mod_json_template_,
                               mod_json_allocator_);
  if (parse_status != JSON_OK) {
    fprintf(stderr, "%s at %zd\n", jsonStrError(parse_status),
            endptr - buffer.data());
    return false;
  }

  return true;
}

bool InferenceJsonGenerator::LoadLLDTemplate(std::string template_path) {
  std::vector<char> buffer;
  if (!ReadFile(template_path, buffer)) {
    std::cerr << "Can not read template file" << std::endl;
    return false;
  }

  char *endptr;
  int parse_status = jsonParse(buffer.data(), &endptr, &lld_json_template_,
                               lld_json_allocator_);
  if (parse_status != JSON_OK) {
    fprintf(stderr, "%s at %zd\n", jsonStrError(parse_status),
            endptr - buffer.data());
    return false;
  }

  return true;
}

bool InferenceJsonGenerator::ReadFile(std::string file_path,
                                      std::vector<char> &buffer) {
  std::ifstream input_stream(file_path.c_str());
  if (!input_stream.is_open()) {
    std::cerr << "Can not open template file" << std::endl;
    return false;
  }

  input_stream.seekg(0, std::ios::end);  // go to the end
  int length = input_stream.tellg();     // report location (this is the length)
  input_stream.seekg(0, std::ios::beg);  // go back to the beginning
  buffer = std::vector<char>(
      length);  // allocate memory for a buffer of appropriate dimension
  input_stream.read(buffer.data(), length);
  input_stream.close();
  return true;
}

}  // namespace perception
}  // namespace xpilot
