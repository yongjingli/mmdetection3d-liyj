cmake_minimum_required(VERSION 3.1)
project(onnx2trt_rcnn)

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD  11)

set(PROTOBUF_LIBRARY ${PROJECT_SOURCE_DIR}/third_party/protobuf/lib/libprotobuf.a)
set(PROTOBUF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf/include)
# set(TENSORRT_ROOT /home/dyg/software/TensorRT-5.0.2.6)
set(TENSORRT_ROOT /home/dyg/software/TensorRT-6.0.1.5)
set(CUDNN_ROOT_DIR /home/dyg/software/cudnn-10.0-linux-x64-v7.6.3)

set(OPENCV_CONFIG /home/dyg/software/opencv-3.4.7/install/usr/local/share/OpenCV/OpenCVConfig.cmake)
if (EXISTS ${OPENCV_CONFIG})
    include(${OPENCV_CONFIG})
else()
    find_package(OpenCV3 REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_CV -DONNX_NAMESPACE=onnx2trt_onnx -DONNX_API=\"\" -fPIC -std=gnu++11") #-O3 -DNDEBUG

#--------------------NVIDIA begin--------------------#
FIND_PACKAGE(CUDA REQUIRED)
list(APPEND GPU_ARCHS
        60  #aliyun
        61  #server1(liao-HP-Z440-Workstation)/server2(herbie)
        62  #TX2/PX2
        70  #server3(liaozl-SYS-7048GR-TR)/Xavier
        #72  #xpu
        )

set(CUDA_VERBOSE_BUILD ON)

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture
list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --std=c++11 \
    -cudart static \
    -lineinfo \
    -g \
    --expt-extended-lambda \
    -Wall -Wno-pointer-arith \
    --compiler-options -fPIC \
    ${GENCODES} \
")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODES})  #add by dyg

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# CUDNN
# set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES cuda/include include)
find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
find_package_handle_standard_args(
        CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

if(NOT CUDNN_FOUND)
    message(WARNING
            "Cudnn cannot be found. TensorRT depends explicitly "
            "on cudnn so you should consider installing it.")
    return()
endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR
            "Cannot find TensorRT library.")
endif()

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

include_directories(${CUDA_INCLUDE_DIRS})
#--------------------NVIDIA end--------------------#

#--------------------onnx-tensorrt begin--------------------#
# Enable compiler warnings
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif()
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W4")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ONNX_NAMESPACE)
    set(ONNX_NAMESPACE "onnx2trt_onnx")
endif()
add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

set(PLUGIN_SOURCES
        onnx-tensorrt/FancyActivation.cu
        onnx-tensorrt/ResizeNearest.cu
        onnx-tensorrt/Split.cu
        onnx-tensorrt/InstanceNormalization.cpp
        onnx-tensorrt/plugin.cpp
        plugin/RoIAlign.cu
        plugin/maskrcnn_roi.cpp
        plugin/topKnms.cu
        plugin/gemvInt8.cu
        plugin/retinanet_nms.cu
        plugin/upSampleInt8.cu
        )
set(IMPORTER_SOURCES
        onnx-tensorrt/NvOnnxParser.cpp
        onnx-tensorrt/ModelImporter.cpp
        onnx-tensorrt/builtin_op_importers.cpp
        onnx-tensorrt/onnx2trt_utils.cpp
        onnx-tensorrt/ShapedWeights.cpp
        onnx-tensorrt/OnnxAttrs.cpp
        onnx-tensorrt/onnx-operators_onnx2trt_onnx.pb.cpp #add by dyg
        onnx-tensorrt/onnx_onnx2trt_onnx.pb.cpp #add by dyg
        )
set(RUNTIME_SOURCES
        onnx-tensorrt/NvOnnxParserRuntime.cpp
        onnx-tensorrt/PluginFactory.cpp
        onnx-tensorrt/builtin_plugins.cpp
        )
set(HEADERS
        NvOnnxParser.h
        NvOnnxParserRuntime.h
        )
include_directories(
        plugin
        onnx-tensorrt
        src
)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

FIND_PACKAGE(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

# --------------------------------
# Plugin library
# --------------------------------
if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
    list(APPEND CUDA_NVCC_FLAGS -std=c++11)
endif()
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC --expt-extended-lambda")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(nvonnxparser_plugin STATIC ${PLUGIN_SOURCES})
target_include_directories(nvonnxparser_plugin PUBLIC ${PROTOBUF_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser_plugin ${TENSORRT_LIBRARY})

# --------------------------------
# Importer library
# --------------------------------
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${PROTOBUF_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser_static PUBLIC nvonnxparser_plugin ${PROTOBUF_LIBRARY} ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})

# --------------------------------
# Runtime library
# --------------------------------
add_library(nvonnxparser_runtime_static STATIC ${RUNTIME_SOURCES})
target_include_directories(nvonnxparser_runtime_static PUBLIC ${PROTOBUF_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(nvonnxparser_runtime_static PUBLIC nvonnxparser_plugin ${CUDNN_LIBRARY} ${TENSORRT_LIBRARY})
#--------------------onnx-tensorrt end--------------------#

include_directories(
        src
        third_party
)

#add_subdirectory(onnx-tensorrt)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(NVNPPLIBS nppc npps nppig nppitc nppif nppist)

ADD_LIBRARY(onnxtrt SHARED src/onnx2trt_lib.cpp src/conv_relu.cpp)
target_link_libraries(onnxtrt ${CUDA_LIBRARIES} ${NVNPPLIBS} nvonnxparser_plugin nvonnxparser_static nvonnxparser_runtime_static)

add_executable(onnx2trt_rcnn src/onnx2trt.cpp)
target_link_libraries(onnx2trt_rcnn onnxtrt)

add_executable(maskrcnn_cpp sample/maskrcnn.cpp)
target_link_libraries(maskrcnn_cpp ${OpenCV_LIBS} dl onnxtrt) # add onnxtrt lib to force IDE rebuild libonnxtrt.so first
