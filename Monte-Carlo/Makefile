
# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.

TARGET = MCHost
TARGET_GPU = MCHost_GPU
TARGET_CPU = MCHost_CPU
TARGET_IOCL = MCHost_IOCL

ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

SRCS = main.cpp
SRCS_FILES = $(foreach F, $(SRCS), host/src/$(F))
COMMON_FILES = ../common/src/AOCL_Utils.cpp

HEADERS = ./include
# arm cross compiler
CROSS-COMPILE = arm-linux-gnueabihf-

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG=$(shell aocl compile-config --arm) -I../common/inc
AOCL_LINK_CONFIG=$(shell aocl link-config --arm)

#GPU COMPILATION
GPU_INCLUDE = "$(CUDA_PATH)\include"
GPU_LIB = "$(CUDA_PATH)\lib\Win32"

#INTEL GPU COMPILATION
IOCL_INCLUDE = "$(INTELOCLSDKROOT)\include"
IOCL_LIB =  "$(INTELOCLSDKROOT)\lib\x86"

#OPENMP COMPILATION
SRCS_CPU = mc_cpu.cpp
SRCS_CPU_FILES = $(foreach F, $(SRCS_CPU), openmp_implementation/$(F))

all :
	$(CROSS-COMPILE)g++ -w -D ALTERA -I $(HEADERS) $(SRCS_FILES) $(COMMON_FILES) -o $(TARGET) $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)

nvidia_gpu :
	g++ $(SRCS_FILES) -I $(GPU_INCLUDE) -I $(HEADERS) -D NVIDIA -L $(GPU_LIB) -o $(TARGET_GPU) -lOpenCL -w

cpu :
	g++ $(SRCS_CPU_FILES) -I $(HEADERS) -O3 -o $(TARGET_CPU) -fopenmp -w

intel_gpu :
	g++ $(SRCS_FILES) -I $(IOCL_INCLUDE) -I $(HEADERS) -D IOCL -L $(IOCL_LIB) -o $(TARGET_IOCL) -lOpenCL -w

clean :
	@rm -f *.o $(TARGET)
	@rm -f *.o $(TARGET_CPU)
	@rm -f *.o $(TARGET_GPU)
