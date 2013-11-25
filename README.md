MemoryBuffer
============

CUDA device/host memory buffer abstraction class in C++

It is basically a simple container for device/host memory array.
The benefit of this class is that copying data among device and host memory is delayed to the moment it is really needed.

For example, when it is created initially with host memory array, the device memory pointer stays NULL. The corresponding device memory is allocated only when something is try to read it's device memory. Then the internal logic finds that host memory contains current content and peforms host to device memory copy.
This is performed transparently to user, all that user needs to do is accessing memory address through addr() method, identifying required memory space and read/write access mode with the arguments.
