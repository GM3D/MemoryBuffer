COMMONFLAGS = -O0 -DNDEBUG -g -pg -Xlinker --demangle -I/usr/local/cuda/include
CPPFLAGS =  $(INCLUDES)
#CXXFLAGS = -Wall -fprofile-arcs -ftest-coverage $(COMMONFLAGS)
CXXFLAGS = -Wall $(COMMONFLAGS)
CCFLAGS = -Wall $(COMMONFLAGS)
CC = g++
NVCC = nvcc
CXX = g++
NVCCFLAGS := $(COMMONFLAGS) --ptxas-options="-v" -lineinfo -arch=sm_21 --use_fast_math
LDFLAGS = -Xlinker --demangle -g -pg -L$(HOME)/lib -L/usr/local/cuda/lib64
LDLIBS =   -lcudart -lUnitTest++ -lgcov

TARGETS = testMemoryBuffer
OBJECTS = testMemoryBuffer.o cudaUtils.o
HEADERS = *.h

all: $(TARGETS)

testMemoryBuffer: $(OBJECTS)

$(OBJECTS): $(HEADERS) *.cpp


.PHONY: clean

clean:
	rm -f $(TARGETS) $(OBJECTS)