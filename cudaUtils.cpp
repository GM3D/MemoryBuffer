#include "cudaUtils.h"

namespace gm3d {
  void check(cudaError e, const char *header)  {
    if(e != cudaSuccess){
      if(header) std::cerr << header << ":";
      std::cerr << cudaGetErrorString(e) << "\n";
    }
  }
};
