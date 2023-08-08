cmake_minimum_required(VERSION 3.1)
project(KPI_evaluator)

set(CMAKE_CXX_STANDARD 11)

include_directories(./
                    ./json
                    ../../pipeline
                    ../../pipeline/inference
                    ../../common/utils
                    )

link_directories(
  ./json
)

set(JSON_SOURCES json/gason.cpp json/json_builder.cpp json/json_formatter.cpp)
set(UTIL_SOURCES ../../common/utils/log_utils.c)

set(SOURCES
    main.cpp
    inference_json_generator.cpp
    #[[../../pipeline/inference/inference_decoder.cpp]]
    ${JSON_SOURCES}
    ${UTIL_SOURCES})

add_executable(KPI_evaluator ${SOURCES})

set(CMAKE_CXX_FLAGS "-I -Wall -std=c++11 -m64 -g -std=c++11")

target_link_libraries(KPI_evaluator)


add_custom_command(
        TARGET KPI_evaluator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/template_pred_mod.txt
                ${CMAKE_CURRENT_BINARY_DIR}/template_pred_mod.txt)
