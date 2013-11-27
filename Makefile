COMMONFLAGS = -O0 -DNDEBUG -g -pg -Xlinker --demangle -I/usr/local/cuda/include
CPPFLAGS =  $(INCLUDES)
CXXFLAGS = -Wall -fprofile-arcs -ftest-coverage $(COMMONFLAGS)
#CXXFLAGS = -Wall $(COMMONFLAGS)
CCFLAGS = -Wall $(COMMONFLAGS)
CC = g++
NVCC = nvcc
CXX = g++
NVCCFLAGS := $(COMMONFLAGS) -lineinfo -arch=sm_21 --use_fast_math
LDFLAGS = -Xlinker --demangle -g -pg -L$(HOME)/lib -L/usr/local/cuda/lib64
LDLIBS =   -lcudart -lUnitTest++ -lgcov

TARGETS = testMemoryBuffer testMemoryBuffer2
OBJECTS0 = cudaUtils.o MemoryBuffer.o MemoryBuffer_cuda.o
OBJECTS1 = testMemoryBuffer.o $(OBJECTS0)
OBJECTS2 = testMemoryBuffer2_cuda.o $(OBJECTS0)
HEADERS = *.h

all: $(TARGETS)

testMemoryBuffer: $(OBJECTS1)
testMemoryBuffer2: $(OBJECTS2)
	g++ $(LDFLAGS) $(OBJECTS2) $(LDLIBS) -o $@

$(OBJECTS): $(HEADERS) *.cpp

%_cuda.o: %.cu #Makefile
	$(NVCC)  $(NVCCFLAGS) -c $*.cu -o $@


.PHONY: clean

clean:
	rm -f $(TARGETS) *.o