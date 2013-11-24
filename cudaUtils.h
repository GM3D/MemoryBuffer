#ifndef CUDAUTILS_H
#define CUDAUTILS_H
#include <iostream>
#include <cuda_runtime.h>

namespace gm3d {
  void check(cudaError e, const char *header=NULL);
};

#endif // CUDAUTILS_H
