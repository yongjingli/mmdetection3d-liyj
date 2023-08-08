#******************************************************************************
#* Copyright (c) 2018 - Present, XMotors.ai, Inc. and its Affiliates.
#*****************************************************************************/
#make third_partys
#make

#Commands for X86, TensorRT-5.0.0.10 or TensorRT-5.0.2.6. GPU5 server has so for TensorRT5.0.2.
#make third_partys ARCH=x86_64 TENSORRTPATH=third_party/TensorRT-5.0.2.6 TENSORRTLIB=/usr/lib/x86_64-linux-gnu/
#make ARCH=x86_64 TENSORRTPATH=third_party/TensorRT-5.0.2.6 TENSORRTLIB=/usr/lib/x86_64-linux-gnu/

ARCH		?=aarch64
ONNXTRTLIB	=onnx-tensorrt
CUDAPATH	=/usr/local/cuda
CUDALIB		=$(CUDAPATH)/targets/$(ARCH)-linux
TENSORRTPATH?=$(CUDALIB)
TENSORRTLIB	?=$(TENSORRTPATH)/lib
INCPATHS	= -Isrc -Iplugin -I./third_party -I./third_party/protobuf/include -I/usr/local/include -I$(ONNXTRTLIB) -I$(CUDALIB)/include -I$(ONNXTRTLIB)/onnx -I/usr/include/aarch64-linux-gnu -I/usr/local/include -I$(TENSORRTPATH)/include
CFLAGS		=  $(INCPATHS) -fopenmp  -DONNX_NAMESPACE=onnx2trt_onnx -DONNX_API=""  -Wall -Wno-deprecated-declarations -Wno-pointer-arith  -O3 -DNDEBUG -fPIC   -std=gnu++11 \
	

LFLAGS1		= -fopenmp -fPIC -ldl -lrt -lcudart -lcudnn -lnvinfer -lnppc -lnppig -lnpps -lnppitc -lnppist -L$(CUDALIB)/lib/ -L$(TENSORRTLIB)
LFLAGS		= $(LFLAGS1) third_party/protobuf/lib/libprotobuf.a
LIB_OPENCV = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

#From QNX version, instead of CMake( used by onnx-tensorrt )
AT=@
AR = ar cr
ECHO = @echo
SHELL=/bin/sh
STRIP=strip

EXTRA_DIRECTORIES	=onnx-tensorrt plugin
OUTNAME_RELEASE 	?= onnx2trt_rcnn
OUTDIR 				?= bin
OBJDIR    			=$(OUTDIR)/chobj
OBJS	=$(patsubst %.cpp, $(OBJDIR)/%.o, $(wildcard *.cpp $(addsuffix /*.cpp, $(EXTRA_DIRECTORIES))))
CUOBJS	=$(patsubst %.cu, $(OBJDIR)/%.o, $(wildcard *.cu $(addsuffix  /*.cu, $(EXTRA_DIRECTORIES))))
OBJS 	+= $(OBJDIR)/src/conv_relu.o $(OBJDIR)/src/onnx2trt.o $(OBJDIR)/src/onnx2trt_lib.o
LIBOBJS = $(CUOBJS) $(OBJDIR)/onnx-tensorrt/NvOnnxParserRuntime.o  $(OBJDIR)/onnx-tensorrt/plugin.o $(OBJDIR)/onnx-tensorrt/InstanceNormalization.o $(OBJDIR)/onnx-tensorrt/PluginFactory.o  $(OBJDIR)/onnx-tensorrt/builtin_plugins.o $(OBJDIR)/plugin/maskrcnn_roi.o $(OBJDIR)/src/conv_relu.o $(OBJDIR)/src/onnx2trt_lib.o
GENCODES = -gencode arch=compute_72,code=sm_72 --default-stream per-thread

CXX		?=g++
CUCC 	?= nvcc -m64 -ccbin $(CXX) 
CUFLAGS += $(INCPATHS) --std=c++11 -cudart static -lineinfo -g --expt-extended-lambda $(GENCODES) --Wno-deprecated-declarations

.PHONY:all release sample lib convert test clean third_party

	
all: release lib liblite sample
	echo "Build All Finished"
	
release : $(OUTDIR)/$(OUTNAME_RELEASE)
	$(ECHO) Build $@ $^ success
	
$(OUTDIR)/$(OUTNAME_RELEASE) : $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	$(AT)$(CXX) -o $@ $^ $(LFLAGS) -Wl,--start-group $(LIBS) -Wl,--end-group

	
samples: $(OUTDIR)/test_models $(OUTDIR)/test_multi_models $(OUTDIR)/test_models_cuda
	$(ECHO) Build $@ $^ success
	
$(OUTDIR)/test_models: sample/maskrcnn.cpp 
	#$(AT)$(CXX) sample/maskrcnn.c -o $(OUTDIR)/maskrcnn_c $(CFLAGS) $(OUTDIR)/libonnxtrt.so
	$(AT)$(CXX) sample/maskrcnn.cpp -o $(OUTDIR)/test_models $(CFLAGS) -O3 -g 	-ldl
	#$(AT)$(CXX) sample/maskrcnn.cpp -o $(OUTDIR)/test_models_cv -DUSE_CV $(CFLAGS) -O3 -g 	-ldl $(LIB_OPENCV)

$(OUTDIR)/test_multi_models: sample/maskrcnn_mulit_models.cpp
	$(AT)$(CXX) sample/maskrcnn_mulit_models.cpp -o $(OUTDIR)/test_multi_models $(CFLAGS) -O3 -g -ldl 
	#$(AT)$(CXX) sample/maskrcnn_mulit_models.cpp -o $(OUTDIR)/test_multi_models_cv $(CFLAGS) -O3 -g -ldl -DUSE_CV $(LIB_OPENCV)

$(OUTDIR)/test_models_cuda: sample/test_models_cuda.cu
	$(AT)$(CUCC) $(CUFLAGS) -o $@ $<

lib: $(OUTDIR)/libonnxtrt.so
	$(ECHO) Build $@ $^ success
		
$(OUTDIR)/libonnxtrt.so: $(LIBOBJS)
	$(ECHO) "Linking: $^ --> $@ $(OUTDIR)/libonnxtrt.so"
	$(ECHO) "{\n global: *Engine;\n local: *;\n };\n" > /tmp/libexport.txt
	$(AT)$(CXX) -shared -Wl,-soname,libonnxtrt.so -fPIC -o $(OUTDIR)/libonnxtrt.so $^ $(LFLAGS1) \
			-Wl,--start-group -Wl,--end-group -Wl,--version-script=/tmp/libexport.txt
	$(AT)${STRIP} --strip-all $(OUTDIR)/libonnxtrt.so
	$(AT)-rm -rf /tmp/libexport.txt
	
convert_maskrcnn:
	#load onnx with tensorrt and save CudaEngine as .trt
	./bin/onnx2trt_rcnn -o parking_masknet.trt -d 16 -b 16 parking_masknet.onnx
	./bin/onnx2trt_rcnn -o parking_keypoint.trt -d 16 -b 32 parking_keypoint.onnx
	cat parking_masknet.trt > masknet_mk_kp.trt
	cat parking_keypoint.trt >> masknet_mk_kp.trt
	TRT_FEAT=4 ./bin/onnx2trt_rcnn -o maskrcnn_fp16b2.trt -m masknet_mk_kp.trt -N 100 -d 16 -b 2 -l maskrcnn.onnx
	#test the speed of CudaEngine
	./bin/onnx2trt_rcnn -o maskrcnn_fp16b2.trt -m masknet_mk_kp.trt -i 32.ppm -p
	
convert_xpmodel:
	#load onnx with tensorrt and save CudaEngine as .trt
	./bin/onnx2trt /tmp/xp_model.onnx  -o mod_net/xp_model_xavier_fp16.trt -b 1 -d 16
	#TRT_IW=960 TRT_IH=604 ./bin/onnx2trt /tmp/xp_model.onnx  -o mod_net/xp_model_xavier_fp16.trt -b 1 -d 16
	#test the speed of CudaEngine
	./bin/onnx2trt -o mod_net/xp_model_xavier_fp16.trt -b 1 -d 16 -p

test:
	- @mkdir logs
	#Test1:libonnxtrt_lite.so support LaneMOD, need xp_model_fp16.trt which converted from xp_model.onnx
	bin/test_models mod_net/xp_model_xavier_fp16.trt bin/libonnxtrt_lite.so mod_net/ab198b75.ppm LaneMOD mod_net/ab198b75_LaneMOD_lite.ppm > logs/TestLog_LaneMOD.txt
	#Test2:libonnxtrt_lite.so support MODNet, need mod_480x302_fp16.trt which converted from mod_net.onnx
	bin/test_models mod_net/mod_480x302_fp16.trt bin/libonnxtrt_lite.so mod_net/ab198b75.ppm MODNet mod_net/ab198b75_MODNet_lite.ppm > logs/TestLog_MODNet.txt
	#Test3:libonnxtrt_lite.so support LaneNet, need lane_net_03112019_2.trt which converted from lane_net.onnx
	bin/test_models lane_net/lane_net_03112019_2.trt bin/libonnxtrt_lite.so lane_net/0a0eef76.ppm LaneNet lane_net/0a0eef76_out.ppm > logs/TestLog_LaneNet.txt
	#Test4:libonnxtrt.so support MaskRCNN, need maskrcnn_fp16b2.trt and masknet_mk_kp.trtn
	 bin/test_models maskrcnn_fp16b2.trt bin/libonnxtrt.so 32.ppm masknet_mk_kp.trt 32_out.ppm > logs/TestLog_MaskRCNN.txt

third_partys: 
	export PATH=$PATH:$TENSORRTPATH
	cd third_party && chmod 777 build_third_party.sh && ./build_third_party.sh && cd ..

clean_third_partys:
	$(ECHO) Cleaning third_partys ...
	$(AT)-rm -rf onnx-tensorrt third_party/protobuf third_party/protobuf-3.5.1

liblite: $(OBJDIR)/src/onnx2trt_lite.o 
	$(ECHO) "Linking: $^ --> $@ $(OUTDIR)/libonnxtrt_lite.so"
	$(ECHO) "{\n global: *Engine;\n local: *;\n };\n" > /tmp/libexport.txt
	$(AT)$(CXX) -shared -Wl,-soname,libonnxtrt_lite.so -fPIC -o $(OUTDIR)/libonnxtrt_lite.so $^  \
			-L$(TENSORRTLIB) -lcudart -lnvinfer -Wl,--version-script=/tmp/libexport.txt
	$(AT)${STRIP} --strip-all $(OUTDIR)/libonnxtrt_lite.so
	$(AT)-rm -rf /tmp/libexport.txt
	

######################################################################### CPP	
$(OBJDIR)/%.o: %.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(AT)if [ ! -d $(OBJDIR)/src ]; then mkdir -p $(OBJDIR)/src; fi
	$(AT)$(foreach XDIR,$(EXTRA_DIRECTORIES), if [ ! -d $(OBJDIR)/$(XDIR) ]; then mkdir -p $(OBJDIR)/$(XDIR); fi;) :
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<


######################################################################### CU
$(OBJDIR)/%.o: %.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(foreach XDIR,$(EXTRA_DIRECTORIES), if [ ! -d $(OBJDIR)/$(XDIR) ]; then mkdir -p $(OBJDIR)/$(XDIR); fi;) :
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling CUDA release: $<
	$(AT)$(CUCC) $(CUFLAGS) -c -o $@ $<
		
clean:
	rm -rf $(OBJDIR) $(OUTDIR)/$(OUTNAME_RELEASE)
