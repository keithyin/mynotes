# CUDA编程（三）：CUDA C RUNTIME

本文将从以下几个方面来介绍 CUDA C RUNTIME：

> cuda c runtime 的实现在 cudart library 中

* device memory
* [Shared Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) 
* [Page-Locked Host Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)
* [Asynchronous Concurrent Execution](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) 
* [Multi-Device System](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)
* [Error Checking](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking) 
* [Call Stack](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack) 
* [Texture and Surface Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory) 
* [Graphics Interoperability](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability)



## device memory

CUDA 编程模型认为 系统由 host 和 device 构成，host 和 device 有他们独立的 memory，所以 runtime 提供了一些 函数用来：分配内存，回收内存，复制数据，host 与 device 之间的数据传输。

`device memory`  可以被分配为 **linear memory** 或者 **cuda arrays**





