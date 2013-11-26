#include <unittest++/UnitTest++.h>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cstring>
#include "datatype.h"
#include "MemoryBuffer.h"

TEST(Sanity)
{
  int writes = 0;
  srand(time(NULL));
  int n = rand() % 1000;
  int initial_space = rand() % 2;
  std::cout << "testing with " << n << " " 
	    << data_type_name << " elements.\n";
  std::vector<TYPE> in(n);
  std::vector<TYPE> out(n);
  
  for(int j = 0; j < n; j++){
    in[j] = (TYPE)(j);
    out[j] = (TYPE)(0);
  }

  gm3d::MemoryBuffer<TYPE>::mem_space s =
    (initial_space == 0) ? 
    gm3d::MemoryBuffer<TYPE>::host:
    gm3d::MemoryBuffer<TYPE>::device;
  gm3d::MemoryBuffer<TYPE> mb0(n, s);

  for(int i = 0; i < 100; i++){
    int space = rand() % 2;
    int direction = rand() % 3;
    int assignment = rand() % 3;
    int copyctor = rand() %2;
    // int assignment = 1;
    // int copyctor = 1;
    if(space == 0){
      /// host space.
      if(direction == 0){
	/// host read
	if(assignment){
	  if(copyctor){
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    mb1 = mb0;
	    gm3d::MemoryBuffer<TYPE> mb2(mb1);
	    memcpy(out.data(), 
		   mb2.addr(gm3d::MemoryBuffer<TYPE>::host,
			    gm3d::MemoryBuffer<TYPE>::read),
		   (size_t)(n * sizeof(TYPE)));

	  }else{
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    mb1 = mb0;
	    memcpy(out.data(), 
		   mb1.addr(gm3d::MemoryBuffer<TYPE>::host,
			    gm3d::MemoryBuffer<TYPE>::read),
		   (size_t)(n * sizeof(TYPE)));
	  }
	}else{
	  if(copyctor){
	    gm3d::MemoryBuffer<TYPE> mb2(mb0);
	    memcpy(out.data(), 
		   mb2.addr(gm3d::MemoryBuffer<TYPE>::host,
			    gm3d::MemoryBuffer<TYPE>::read),
		   (size_t)(n * sizeof(TYPE)));
	  }else{	  
	    memcpy(out.data(), 
		   mb0.addr(gm3d::MemoryBuffer<TYPE>::host,
			    gm3d::MemoryBuffer<TYPE>::read),
		   (size_t)(n * sizeof(TYPE)));
	  }
	}
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
	if(assignment){
	  if(copyctor){
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    memcpy(mb1.addr(gm3d::MemoryBuffer<TYPE>::host, 
			    gm3d::MemoryBuffer<TYPE>::write),
		   in.data(),
		   (size_t)(n * sizeof(TYPE)));
	    gm3d::MemoryBuffer<TYPE> mb2(mb1);
	    mb0 = mb2;
	  }else{
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    memcpy(mb1.addr(gm3d::MemoryBuffer<TYPE>::host, 
			    gm3d::MemoryBuffer<TYPE>::write),
		   in.data(),
		   (size_t)(n * sizeof(TYPE)));
	    mb0 = mb1;
	  }
	}else{
	  memcpy(mb0.addr(gm3d::MemoryBuffer<TYPE>::host, 
			  gm3d::MemoryBuffer<TYPE>::write),
		 in.data(),
		 (size_t)(n * sizeof(TYPE)));
	}
      }
    }else if(space == 1){
      /// device space.
      if(direction == 0){
	/// device read
	if(assignment){
	  if(copyctor){
	    gm3d::MemoryBuffer<TYPE> mb1(mb0);
	    gm3d::MemoryBuffer<TYPE> mb2 = mb1;
	    gm3d::check(cudaMemcpy(out.data(),
				   mb2.addr(gm3d::MemoryBuffer<TYPE>::device,
					    gm3d::MemoryBuffer<TYPE>::read),
				   (size_t)(n * sizeof(TYPE)),
				   cudaMemcpyDeviceToHost));
	  }else{
	    gm3d::MemoryBuffer<TYPE> mb2 = mb0;
	    gm3d::check(cudaMemcpy(out.data(),
				   mb2.addr(gm3d::MemoryBuffer<TYPE>::device,
					    gm3d::MemoryBuffer<TYPE>::read),
				   (size_t)(n * sizeof(TYPE)),
				   cudaMemcpyDeviceToHost));
	  }
	}
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
	if(assignment){
	  if(copyctor){
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    gm3d::check(cudaMemcpy(mb1.addr(gm3d::MemoryBuffer<TYPE>::device, 
					    gm3d::MemoryBuffer<TYPE>::write),
				   in.data(),
				   (size_t)(n * sizeof(TYPE)),
				   cudaMemcpyHostToDevice));
	    gm3d::MemoryBuffer<TYPE> mb2(mb1);
	    mb0 = mb2;
	  }else{
	    gm3d::MemoryBuffer<TYPE> mb1(n);
	    gm3d::check(cudaMemcpy(mb1.addr(gm3d::MemoryBuffer<TYPE>::device, 
					    gm3d::MemoryBuffer<TYPE>::write),
				   in.data(),
				   (size_t)(n * sizeof(TYPE)),
				   cudaMemcpyHostToDevice));
	    mb0 = mb1;
	  }
	}else{
	  gm3d::check(cudaMemcpy(mb0.addr(gm3d::MemoryBuffer<TYPE>::device, 
					  gm3d::MemoryBuffer<TYPE>::write),
				 in.data(),
				 (size_t)(n * sizeof(TYPE)),
				 cudaMemcpyHostToDevice));
	}
      }
    }
    if(space >= 2 || direction >= 2){
      gm3d::MemoryBuffer<TYPE>::mem_space s = 
	(gm3d::MemoryBuffer<TYPE>::mem_space)space;
      gm3d::MemoryBuffer<TYPE>::access_mode m = 
	(gm3d::MemoryBuffer<TYPE>::access_mode)direction;
      /// test exception only once per 20 loops.
      int chance = rand() % 100;
      if(chance < 95) continue;
      std::cout << "trying invalid combination: (" 
		<< space << ", " << direction << ")\n";
      TYPE *ptr = NULL;
      std::cout << "checking exception: (i = " << i << ")\n";
      CHECK_THROW(ptr = mb0.addr(s,m), std::invalid_argument);
      std::cout << "ptr = " << ptr << "\n";
    }
  }
}



int main(int argc, char **argv)
{
  return UnitTest::RunAllTests();
}
