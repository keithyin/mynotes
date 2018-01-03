# GPU 编程： CUDA



## 一个典型的 cuda 程序

* CPU allocate storage on GPU (`cudaMalloc()`) 
  * (dynamic parallelism) GPU can also allocate storage on GPU supported by dynamic parallelism
* CPU copies input data from CPU to GPU (`cudaMemcpy`)
* CPU launches `kernels` on GPU to Process data (`kernel launch`)
  * (dynamic parallelism) GPU can also launch `kernels` supported by dynamic parallelism
* CPU copies result back to CPU from GPU (`cudaMemcpy`)
* release the GPU resources
  * (dynamic parallelism) 如果是 kernel 分配的 内存，应该也是由 kernel 来 释放空间。



从上面可以看出 **CPU is the BOSS !!!**

 

**关于 CUDA**

* CUDA  假设 GPU 是 CPU 的 协处理器。
* GPU 称为 `device`， CPU 端称为 `host`
* CUDA 同样假设： CPU 和 GPU 的 内存空间是 独立的 。
* `host launches kernels on device`。



##  DL 框架中如何 组织 代码

* GPU 上跑的代码都写在 `.cu` 文件中， 
  * `__global__, __device__` 
* 然后写个 `launcher` 来调用 `kernel`
  * `launcher` **仅仅用来 调用 `kernel`**
  * 计算 `gridDim， blockDim`
  * 然后将 `launcher` 接口暴露给其它 应用就可以了。

**`.cu` 代码只需要搞定上面两个就可以了**



**剩下就是 `.cc/.c` 代码调用 `launcher` 了。  **



