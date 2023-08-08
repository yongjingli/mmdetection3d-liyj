/*
 * json_formater.h
 *
 *  Created on: Apr 17, 2019
 *      Author: reza
 */

#ifndef COMMON_JSON_JSON_FORMATTER_H_
#define COMMON_JSON_JSON_FORMATTER_H_

#include <sstream>
#include <string.h>

#include "gason.h"

namespace xpilot {
namespace perception {

class JsonFormatter {
public:
    static std::string GetFormattedString(JsonValue o);
private:
    static void dumpString(std::stringstream &stream, const char *s);
    static void dumpValue(std::stringstream &stream, JsonValue o, int indent = 0);
    static void printError(const char *filename, int status, char *endptr, char *source, size_t size);
};


} // namespace perception
} // namespace xpilot

#endif /* COMMON_JSON_JSON_FORMATTER_H_ */
