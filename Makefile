COMMONFLAGS = -O0 -DDEBUG -g -Xlinker --demangle -I/usr/local/cuda/include
CPPFLAGS =  $(INCLUDES)
CXXFLAGS = -Wall $(COMMONFLAGS)
CCFLAGS = -Wall $(COMMONFLAGS)
CC = g++
NVCC = nvcc
CXX = g++
NVCCFLAGS := $(COMMONFLAGS) --ptxas-options="-v" -lineinfo -arch=sm_21 --use_fast_math
LDFLAGS = -Xlinker --demangle -g -pg -L$(HOME)/lib -L/usr/local/cuda/lib64
LDLIBS =   -lcudart -lUnitTest++

TARGETS = testMemoryBuffer
OBJECTS = testMemoryBuffer.o cudaUtils.o
HEADERS = *.h

all: $(TARGETS)

testMemoryBuffer: $(OBJECTS)

$(OBJECTS): $(HEADERS)


.PHONY: clean

clean:
	rm -f $(TARGETS) $(OBJECTS)