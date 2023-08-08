/*
 * json_formater.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: reza
 */

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "json_formatter.h"

namespace xpilot {
namespace perception {

const int SHIFT_WIDTH = 4;
bool debug_enable = false;

std::string JsonFormatter::GetFormattedString(JsonValue o) {
    std::stringstream stream;
    dumpValue(stream, o);
    return stream.str();
}
void JsonFormatter::dumpString(std::stringstream &stream, const char *s) {
//    fputc('"', stdout);
    stream << "\"";
    while (*s) {
        char c = *s++;
        switch (c) {
        case '\b':
//            fprintf(stdout, "\\b");
            stream << "\\b";
            break;
        case '\f':
//            fprintf(stdout, "\\f");
            stream << "\\f";
            break;
        case '\n':
//            fprintf(stdout, "\\n");
            stream << "\\n";
            break;
        case '\r':
            fprintf(stdout, "\\r");
            stream << "\\r";
            break;
        case '\t':
//            fprintf(stdout, "\\t");
            stream << "\\t";
            break;
        case '\\':
            fprintf(stdout, "\\\\");
            stream << "\\\\";
            break;
        case '"':
//            fprintf(stdout, "\\\"");
            stream << "\\\"";
            break;
        default:
//            fputc(c, stdout);
            stream << c;
        }
    }
//    fprintf(stdout, "%s\"", s);
    stream << s << "\"";
}

void JsonFormatter::dumpValue(std::stringstream &stream, JsonValue o, int indent) {
    switch (o.getTag()) {
    case JSON_NUMBER:
//        fprintf(stdout, "%f", o.toNumber());
        stream << o.toNumber();
        break;
    case JSON_STRING:
        dumpString(stream, o.toString());
        break;
    case JSON_ARRAY:{
        // It is not necessary to use o.toNode() to check if an array or object
        // is empty before iterating over its members, we do it here to allow
        // nicer pretty printing.
        if (!o.toNode()) {
//            fprintf(stdout, "[]");
            stream << "[]";
            break;
        }
//        fprintf(stdout, "[\n");
        stream << "[\n";
        for (auto i : o) {
//            fprintf(stdout, "%*s", indent + SHIFT_WIDTH, "");
            for ( int  i = 0 ; i < indent + SHIFT_WIDTH ; i++ )
                stream << " ";
            dumpValue(stream, i->value, indent + SHIFT_WIDTH);
//            fprintf(stdout, i->next ? ",\n" : "\n");
            stream << (i->next ? ",\n" : "\n");
        }
//        fprintf(stdout, "%*s]", indent, "");
        for ( int  i = 0 ; i < indent; i++ )
            stream << " ";
        stream << "]";
        break;
    }
    case JSON_OBJECT:
    {
        if (!o.toNode()) {
//            fprintf(stdout, "{}");
            stream << "{}";
            break;
        }
//        fprintf(stdout, "{\n");
        stream << "{\n";
        for (auto i : o) {
//            fprintf(stdout, "%*s", indent + SHIFT_WIDTH, "");
            for ( int  i = 0 ; i < indent + SHIFT_WIDTH ; i++ )
                stream << " ";
            dumpString(stream, i->key);
//            fprintf(stdout, ": ");
            stream << ": ";
            dumpValue(stream, i->value, indent + SHIFT_WIDTH);
//            fprintf(stdout, i->next ? ",\n" : "\n");
            stream << (i->next ? ",\n" : "\n");
        }
//        fprintf(stdout, "%*s}", indent, "");
        for ( int  i = 0 ; i < indent; i++ )
            stream << " ";
        stream << "}";
        break;
    }
    case JSON_TRUE:
//        fprintf(stdout, "true");
        stream << "true";
        break;
    case JSON_FALSE:
//        fprintf(stdout, "false");
        stream << "false";
        break;
    case JSON_NULL:
//        fprintf(stdout, "null");
        stream << "null";
        break;
    }
}

void JsonFormatter::printError(const char *filename, int status, char *endptr, char *source, size_t size) {
    char *s = endptr;
    while (s != source && *s != '\n')
        --s;
    if (s != endptr && s != source)
        ++s;

    int lineno = 0;
    for (char *it = s; it != source; --it) {
        if (*it == '\n') {
            ++lineno;
        }
    }

    int column = (int)(endptr - s);

    fprintf(stderr, "%s:%d:%d: %s\n", filename, lineno + 1, column + 1, jsonStrError(status));

    while (s != source + size && *s != '\n') {
        int c = *s++;
        switch (c) {
        case '\b':
            fprintf(stderr, "\\b");
            column += 1;
            break;
        case '\f':
            fprintf(stderr, "\\f");
            column += 1;
            break;
        case '\n':
            fprintf(stderr, "\\n");
            column += 1;
            break;
        case '\r':
            fprintf(stderr, "\\r");
            column += 1;
            break;
        case '\t':
            fprintf(stderr, "%*s", SHIFT_WIDTH, "");
            column += SHIFT_WIDTH - 1;
            break;
        case '\0':
            fprintf(stderr, "\"");
            break;
        default:
            fputc(c, stderr);
        }
    }

    fprintf(stderr, "\n%*s\n", column + 1, "^");
}

} // namespace perception
} // namespace xpilot
