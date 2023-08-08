#!/bin/bash
#build third_party for onnx2trt
#sudo apt-get install cmake
#sudo apt-get install build-essential libevent-pthreads-2.1-6
#Step1: protobuf 3.5.1
#wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz
if [ -x "protobuf/bin/protoc" ]
then 
	echo "Find protobuf/bin/protoc"
	cd protobuf 
	export PROTOBUFPATH=$PWD
else  
	tar zxvf protobuf-cpp-3.5.1.tar.gz
	cd protobuf-3.5.1
	if [ "$CROSS_COMPLE" = "" ]  	
	then
		./configure --disable-shared   CFLAGS="-fPIC" CXXFLAGS="-fPIC"  
    else
    	./configure --with-protoc=/usr/local/bin/protoc --disable-shared  --host=arm CC=${CROSS_COMPLE}gcc CXX=${CROSS_COMPLE}g++ LD=${CROSS_COMPLE}ld CFLAGS="-Wall -D_QNX_SOURCE -fPIC" CXXFLAGS="-Wall -std=gnu++11 -stdlib=libstdc++ -D_QNX_SOURCE -fPIC" 
    fi
    
	make -j 4
	#make check -j 4
	#make install
	mkdir ../protobuf 
	cd ../protobuf 
	export PROTOBUFPATH=$PWD
	export PATH=$PWD/bin:$PATH
	protoc --version
	mkdir bin 
	ln -s ../../protobuf-3.5.1/src/protoc bin/protoc 
	ln -s ../protobuf-3.5.1/src/.libs lib
	mkdir include
	ln -s ../../protobuf-3.5.1/src/google/ include/google
fi	

#Step2: onnx-tensorrt & onnx
cd ../
#Step2: onnx-tensorrt & onnx
if [ -e "../onnx-tensorrt" ]
then 
	echo "Find onnx-tensorrt"
else 
	tar zxvf onnx-tensorrt-5.0.3-qnx.tar.gz -C ../
fi

if [ -f "protobuf/bin/protoc" ]
then 
	echo "Build third_party success"
else
	echo "Build third_party faild"
fi

