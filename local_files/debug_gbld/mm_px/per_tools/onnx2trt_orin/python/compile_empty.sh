cd src
/usr/bin/aarch64-linux-gnu-g++ onnx2trt_empty.cpp --shared -o libonnxtrt.so -Wall -Wno-deprecated-declarations -Wno-pointer-arith -std=c++14 -lc -O3 -DNDEBUG -fPIC -Wl,-soname,libonnxtrt.so
