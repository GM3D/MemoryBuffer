#include "MemoryBuffer.h"
#include <unittest++/UnitTest++.h>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cstring>
#include "datatype.h"

TEST(Sanity)
{
  int writes = 0;
  srand(time(NULL));
  int n = rand() % 100;
  std::cout << "testing with " << n << " " 
	    << data_type_name << " elements.\n";
  std::vector<TYPE> in(n);
  std::vector<TYPE> out(n);
  
  for(int j = 0; j < n; j++){
    in[j] = (TYPE)(j);
    out[j] = (TYPE)(0);
  }

  for(int i = 0; i < 1000; i++){
    int initial_space = rand() % 2;
    gm3d::MemoryBuffer<TYPE>::mem_space s =
      (initial_space == 0) ? 
      gm3d::MemoryBuffer<TYPE>::host:
      gm3d::MemoryBuffer<TYPE>::device;
    std::cout << "initial memory space = " << s << "\n";
    gm3d::MemoryBuffer<TYPE> mb(n, s);
    int space = rand() % 3;
    int direction = rand() % 3;
    if(space == 0){
      /// host space.
      if(direction == 0){
	/// host read
	memcpy(out.data(), 
	       mb.addr(gm3d::MemoryBuffer<TYPE>::host,
		       gm3d::MemoryBuffer<TYPE>::read),
	       (size_t)(n * sizeof(TYPE)));
	/// if there was a write prior to this read, confirm
	/// data read out is same as what was written.
	if(writes > 0){
	  for(int j = 0; j < n; j++){
	    CHECK_EQUAL(in[j], out[j]);
	  }
	}
      }else{
	/// host write
	for(int j = 0; j < n; j++){
	  in[j] = (TYPE)(rand());
	}
	memcpy(mb.addr(gm3d::MemoryBuffer<TYPE>::host, 
		       gm3d::MemoryBuffer<TYPE>::write),
	       in.data(),
	       (size_t)(n * sizeof(TYPE)));
      }
    }else if(space == 1){
      /// device space.
      if(direction == 0){
	/// device read
	gm3d::check(cudaMemcpy(out.data(),
			       mb.addr(gm3d::MemoryBuffer<TYPE>::device,
				       gm3d::MemoryBuffer<TYPE>::read),
			       (size_t)(n * sizeof(TYPE)),
			       cudaMemcpyDeviceToHost));
	if(writes > 0){
	  for(int j = 0; j < n; j++){
	    CHECK_EQUAL(in[j], out[j]);
	  }
	}
      }else{
	/// device write
	for(int j = 0; j < n; j++){
	  in[j] = (TYPE)(rand());
	}
	gm3d::check(cudaMemcpy(mb.addr(gm3d::MemoryBuffer<TYPE>::device, 
				       gm3d::MemoryBuffer<TYPE>::write),
			       in.data(),
			       (size_t)(n * sizeof(TYPE)),
			       cudaMemcpyHostToDevice));
      }
    }
    if(space >= 2 || direction >= 2){
      gm3d::MemoryBuffer<TYPE>::mem_space s = 
	(gm3d::MemoryBuffer<TYPE>::mem_space)space;
      gm3d::MemoryBuffer<TYPE>::access_mode m = 
	(gm3d::MemoryBuffer<TYPE>::access_mode)direction;
      std::cout << "trying invalid combination: (" 
		<< space << ", " << direction << ")\n";
      TYPE *ptr;
      try {
	ptr = mb.addr(s, m);
      }
      catch(std::invalid_argument e){
	std::cout << "std::invalid_argument raised.\n";
	std::cout << "ptr = " << ptr << "\n";
      }
      catch(std::domain_error e){
	std::cout << "std::domain_error raised.\n" << e.what();
	std::cout << "ptr = " << ptr << "\n";
      }
      catch(...){
	std::cout << "some unexpected exception raised.\n";
	std::cout << "ptr = " << ptr << "\n";
      }
    }
  }
}

int main(int argc, char **argv)
{
  return UnitTest::RunAllTests();
}
