COMMONFLAGS = -O0 -DNDEBUG -g -pg -Xlinker --demangle -I/usr/local/cuda/include
CPPFLAGS =  $(INCLUDES)
CXXFLAGS = -Wall -fPIC -fprofile-arcs -ftest-coverage $(COMMONFLAGS)
CCFLAGS = -Wall $(COMMONFLAGS)
CC = g++
NVCC = nvcc
CXX = g++
NVCCFLAGS := $(COMMONFLAGS) --compiler-options="-fPIC"\
 	-lineinfo -arch=sm_21 --use_fast_math
LDFLAGS = -Xlinker --demangle -g -pg -L$(HOME)/lib -L/usr/local/cuda/lib64
LDLIBS =   -lcudart -lUnitTest++ -lgcov

TARGETS = testMemoryBuffer testMemoryBuffer2
LIBTARGET = libMemoryBuffer.o
OBJECTS0 = cudaUtils.o MemoryBuffer.o MemoryBuffer_cuda.o
OBJECTS1 = testMemoryBuffer.o $(OBJECTS0)
OBJECTS2 = testMemoryBuffer2_cuda.o $(OBJECTS0)
HEADERS = MemoryBuffer.h cudaUtils.h

all: $(TARGETS)

testMemoryBuffer: $(OBJECTS1)
testMemoryBuffer2: $(OBJECTS2)
	g++ $(LDFLAGS) $(OBJECTS2) $(LDLIBS) -o $@


cudaUtils.o: cudaUtils.h Makefile
MemoryBuffer.o: MemoryBuffer.h cudaUtils.h Makefile
MemoryBuffer_cuda.o: MemoryBuffer.h cudaUtils.h Makefile
testMemoryBuffer.o: datatype.h MemoryBuffer.h Makefile
testMemoryBuffer2_cuda.o: datatype.h MemoryBuffer.h Makefile


%_cuda.o: %.cu
	$(NVCC)  $(NVCCFLAGS) -c $*.cu -o $@


.PHONY: clean
clean:
	rm -f $(TARGETS) *.o

.PHONY: lib
lib: $(OBJECTS0)
	$(CC) -lm -shared -Wl,-soname,$(LIBTARGET) -o $(LIBTARGET) $(OBJECTS0)

.PHONY: install
install: $(LIBTARGET) $(HEADERS)
	install --mode=644 $(LIBTARGET) $(HOME)/lib
	install --mode=644 $(HEADERS) $(HOME)/include

