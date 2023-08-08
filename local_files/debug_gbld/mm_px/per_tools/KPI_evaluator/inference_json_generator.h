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

#ifndef _KPI_INFERENCE_JSON_GENERATOR_H_
#define _KPI_INFERENCE_JSON_GENERATOR_H_

#include <string>
#include <vector>

#include "gason.h"
#include "json_builder.h"
#include "json_formatter.h"

namespace xpilot {
namespace perception {

class InferenceJsonGenerator {
 public:
  InferenceJsonGenerator(std::string mod_template_path,
                         std::string lld_template_path);
  bool GenerateMODJson(std::string output_path);
  bool GenerateLLDJson(std::string output_path);

 private:
  bool LoadMODTemplate(std::string template_path);
  bool LoadLLDTemplate(std::string template_path);
  bool ReadFile(std::string file_path, std::vector<char> &buffer);

  JsonAllocator mod_json_allocator_, lld_json_allocator_;
  JsonValue mod_json_template_, lld_json_template_;
};

}  // namespace perception
}  // namespace xpilot

#endif  // _KPI_INFERENCE_JSON_GENERATOR_H_
