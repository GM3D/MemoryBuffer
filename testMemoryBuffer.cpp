#include "MemoryBuffer.h"
#include <unittest++/UnitTest++.h>

TEST(MemBufSanity)
{
  gm3d::MemoryBuffer<int> mem;
  gm3d::MemoryBuffer<double> memDouble(2000);
  CHECK(true);
}

int main(int argc, char **argv)
{
  return UnitTest::RunAllTests();
}
 

