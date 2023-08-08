
NV_PLATFORM_OS     = QNX
USERHOMEDIR   = 
NV_PLATFORM_LDFLAGS   =
NV_PLATFORM_SDK_LIB_DIR = $(TOPDIR)/platform/drive-t186ref-qnx/lib-target
NV_PLATFORM_SDK_INC   = -I$(TOPDIR)/platform/drive-t186ref-qnx/include \
                        -I$(TOPDIR)/platform/cuda-qnx64-native/include
                        
NV_PLATFORM_SDK_LIB   = -L${USERHOMEDIR}/opt/toolchains/targetfs/usr/libnvidia \
                        -L${USERHOMEDIR}/opt/toolchains/cuda-10.0/targets/aarch64-qnx/lib \
                        -L${USERHOMEDIR}/opt/toolchains/targetfs/usr/local/cuda-10.1/targets/aarch64-qnx/lib \
                        -L$(NV_PLATFORM_SDK_LIB_DIR) \
                        -L$(TOPDIR)/platform/cuda-qnx64-native/lib \
                        -L$(NV_PLATFORM_SDK_LIB_DIR)/$(NV_WINSYS) \
                        -Wl,-rpath-link=$(NV_PLATFORM_SDK_LIB_DIR) \
                        -Wl,-rpath=${USERHOMEDIR}/opt/toolchains/cuda-10.0/targets/aarch64-qnx/lib

#LDFLAGS   = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)
#LDFLAGS  += -L${USERHOMEDIR}/opt/toolchains/cuda-10.0/targets/aarch64-qnx/lib
#LDFLAGS  += -L${USERHOMEDIR}/opt/toolchains/targetfs/usr/local/cuda-10.0/lib

#LDFLAGS  += -L${USERHOMEDIR}/opt/toolchains/targetfs/usr/libnvidia
#LDFLAGS  += -L${USERHOMEDIR}/opt/toolchains/targetfs/usr/local/cuda-10.1/targets/aarch64-qnx/lib
#LDFLAGS  += -L${USERHOMEDIR}/opt/toolchains/cuda-10.0/targets/aarch64-qnx/lib
#LDFLAGS   += -L$(TOPDIR)/cuda-qnx64-native/lib


NV_PLATFORM_OPT       = -Os
NV_PLATFORM_MATHLIB   = -lm
NV_PLATFORM_THREADLIB = -lpthread


QNX_BASE ?= ${USERHOMEDIR}/opt/toolchains/qnx700
QNX_KERNEL_VERSION := 7.0.0
QNX_COMPILER_VERSION := 5.4.0

export QNX_HOST = $(QNX_BASE)/host/linux/x86_64
export QNX_ARCH = aarch64le

ifeq ($(QNX_ARCH), aarch64le)
    export QNX_TARGET = $(QNX_BASE)/target/qnx7
    QNX_TOOLS_VARIANT       := aarch64-unknown-nto-qnx$(QNX_KERNEL_VERSION)
else
    export QNX_TARGET = $(QNX_BASE)/target/qnx6
    QNX_TOOLS_VARIANT := arm-unknown-nto-qnx$(QNX_KERNEL_VERSION)eabi
endif
CROSSBIN = $(QNX_HOST)/usr/bin/$(QNX_TOOLS_VARIANT)-

CC     = $(CROSSBIN)gcc
CXX    = $(CROSSBIN)g++
AR     = $(CROSSBIN)ar
RANLIB = $(CROSSBIN)ranlib
STRIP  = $(CROSSBIN)strip
LD := $(if $(wildcard *.cpp),$(CXX),$(CC))

ifeq ($(QNX_ARCH), aarch64le)
    NV_ARCH_PLATFORM_CPP_FLAGS :=
    NV_ARCH_PLATFORM_CFLAGS    := -fno-builtin-memcpy       \
                                  -finline-limit=300
endif

NV_PLATFORM_CPPFLAGS  = $(NV_ARCH_PLATFORM_CPP_FLAGS) \
                        -DWIN_INTERFACE_CUSTOM \
                        -DNV_GLES_VER_MAJOR=2 \
                        -D_FILE_OFFSET_BITS=64 \
                        -D_QNX_SOURCE \
                        -D_POSIX_C_SOURCE=200112L \
                        -DNV_EMBEDDED_BUILD
                        
NV_PLATFORM_CFLAGS    = $(NV_ARCH_PLATFORM_CFLAGS) \
                        -g                        \
                        -O2                       \
                        -fno-strict-aliasing      \
                        -fno-common               \
                        -fomit-frame-pointer      \
                        -finline-functions        \
                        -fpic                     \
                        -ftree-vectorize          \
                        -Wall                     \
                        -Wcast-align
                        

NV_PLATFORM_CPPFLAGS += -I$(QNX_HOST)/usr/lib/gcc/$(QNX_TOOLS_VARIANT)/$(QNX_COMPILER_VERSION)/include \
                        -I$(QNX_TARGET)/usr/include/c++/$(QNX_COMPILER_VERSION) \
                        -I$(QNX_TARGET)/usr/include/c++/$(QNX_COMPILER_VERSION)/$(QNX_TOOLS_VARIANT) \
                        -I$(QNX_TARGET)/usr/include/c++/$(QNX_COMPILER_VERSION)/backward \
                        -I$(QNX_TARGET)/usr/include \
                        $(NV_PLATFORM_SDK_INC)
                        
NV_PLATFORM_LDFLAGS += -B$(QNX_TARGET)/$(QNX_ARCH)/lib           \
                       -B$(QNX_HOST)/usr/$(QNX_TOOLS_VARIANT)/bin \
                       -Xlinker -nostdlib                        \
                       -Xlinker -rpath-link -Xlinker $(TOPDIR)/lib \
                       -L$(QNX_TARGET)/$(QNX_ARCH)/lib           \
                       -L$(QNX_TARGET)/$(QNX_ARCH)/lib/gcc/$(QNX_COMPILER_VERSION) \
                       -L$(QNX_TARGET)/$(QNX_ARCH)/usr/lib


