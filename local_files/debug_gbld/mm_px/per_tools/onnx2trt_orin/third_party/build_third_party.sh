#!/bin/bash
#build third_party for onnx2trt
#sudo apt-get install cmake
#sudo apt-get install build-essential libevent-pthreads-2.1-6
#Step1: protobuf 3.5.1
#wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz
export PROTOBUFNAME=protobuf
if [ "$CROSS_COMPLE" = "" ]  	
then
  export PROTOBUFNAME=protobuf_host
fi
 
if [ -x "protobuf/bin/protoc" ]
then 
	echo "Find protobuf/bin/protoc"
	cd $PROTOBUFNAME 
	export PROTOBUFPATH=$PWD
else  
	tar zxvf protobuf-cpp-3.5.1.tar.gz
	cd protobuf-3.5.1
	if [ "$CROSS_COMPLE" = "" ]  	
	then
		./configure --disable-shared   CFLAGS="-fPIC" CXXFLAGS="-fPIC"  
    else
    	./configure --with-protoc=../protobuf_host/bin/protoc --disable-shared  --host=arm CC=${CROSS_COMPLE}gcc CXX=${CROSS_COMPLE}g++ LD=${CROSS_COMPLE}ld CFLAGS="-Wall -fPIC" CXXFLAGS="-Wall -std=gnu++11 -fPIC" 
    fi
    
	make -j 4
	make check -j 4
	#make install
  cd ..
	mkdir $PROTOBUFNAME  
	cd $PROTOBUFNAME 
	export PROTOBUFPATH=$PWD
	export PATH=$PWD/bin:$PATH
	mkdir bin lib include 
	mv ../protobuf-3.5.1/src/protoc bin/
	mv ../protobuf-3.5.1/src/.libs/* lib/
	mv ../protobuf-3.5.1/src/google include/
	../protobuf_host/bin/protoc --version
  rm -rf ../protobuf-3.5.1
fi	

#Step2: onnx-tensorrt & onnx
cd ../
if [ -e "../onnx-tensorrt" ]
then 
	echo "Find onnx-tensorrt"
else 
	tar zxvf onnx-tensorrt-5.0.3-qnx.tar.gz -C ../
fi

#Step3: cutlass
if [ -e "cutlass" ]
then
        echo "Find cutlass"
else
        tar zxf cutlass-2.8.0.tar.gz -C ./
fi


if [ -f "protobuf/bin/protoc" ]
then 
	echo "Build third_party success"
else
	echo "Build third_party faild"
fi

