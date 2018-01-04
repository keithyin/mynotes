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



## Parallel Communication Patterns

> * task : 一个 thread 看作一个 task

* map :  tasks read from and write to specific data elements
  * one to one : one input --> one output， 一个输入只影响一个输出
* gather : 
  * many to one: 一个输出和 多个输入元素相关， 或者，**相邻的输出，是由不相邻的输入得到**。
* scatter:
  * one to many: 一个输入 影响多个 输出或者， 相邻的输入 的输出 分散开来。
* stencil:  >=2D data access
  * data reuse
  * 每个输出位置 都会被计算
* transpose:
* reduce
  * all to one
* scan:
  * all to all
* sort
  * all to all



**Array of structures(AoS):** 

**structure of arrays (SoA):**  



## Programming Model

**Programmer:**

* divide up your program into smaller computations called kernels(c/c++ function).
* thread bocks : **group of threads that co-operate to solve a (sub)problem.**
* 可以用 多个 thread blocks （每个 thread blocks 中包含多个 threads）



**Thread Blocks and GPU hardware**

* GPU 有很多 streaming multiprocessors (SMs)
  * 不同 GPU， SMs 的数量不一样
* 一个 SM 中 
  * 有 **很多简单的 processors** , 可以并行执行大量线程
  * 还有 memory
  * 所有的 SMs 独立 并行的执行。
* GPU 负责给 thread blocks 指定 SMs
* **不同 thread blocks 之间不应该存在 合作关系！！**
  * block 之间不存在 通信。



**在CUDA编程中**

* 程序员 只需要 执行 用几个 thread blocks， 每个 block 中有多少 thread
* 然后由 GPU 来调度 哪个 SM 应该 运行哪个 thread block。何时运行 thread block。

**CUDA可以保证**

* 一个 block 内的 所有 thread 在同一时刻 ，同一个 SM 上运行
* 在 下一个 kernel 的 block 运行之前， 当前的 kernel 的所有 blocks 均已执行完毕。（**同 stream**）



## Memory Model

**GPU**

* **local memory** : thread 的私有变量，只能 当前 thread 能够访问
* **shared memory** :  block 中的 所有 thread 可以访问。 shared memory 是对 block 内 threads 共享的。 这个是 SM 上的 memory。
* **global memory** :   所有线程都可以访问，所有 kernel 都可以访问。



**CPU**

* host memory: CUDA 也可以直接访问 host memory。（HOW？？？？）



**local memory**

```c++
__global__ void use_local_memory_gpu(float in){ // in 也在 local memory 上。
  float f; // 在 local memory 上。
}
```

**global memory**

```c++
__global__ void use_global_memory(float *array){
  // array 在local memory 上，但是它指向 global memoey
  	float val = array[threadIdx.x]; // 访问 global memory
}
```

**shared memory**

```c++
// 第一种方法
__global__ void use_shared_memory(float *array){
	int i = threadIdx.x;
    __shared__ float sh_arr[128]; // 分配 shared memory, block 内的线程都可以访问
    sh_arr[i] = array[i];
  	__syncthreads();
  
}

```







## 同步

**what if a thread reads a result before another thread writes it?**

* block 内: **barrier** : points in the program where threads stop and wait, when all threads have reached the barrier, they can proceed.
  * `__syncthreads()`
  * 如何确定是不是应该放 `barrier`， 应该看看`write， read` 操作。
* `cudaDeviceSynchronize();` 
* **内存原子操作**
  * `atomicAdd(&v, 1)`: 原子操作。
  * 原子操作有一系列的 劣势：
    * 只有 **部分** 操作 和 类型支持 .
    * 没有 顺序 保证
    * 慢 **slow**