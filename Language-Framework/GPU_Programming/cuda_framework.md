# 硬件结构与CUDA结构

**GPU Framework( 硬件)**

* Graphic Processing Unit (GPU)
* Graphic Processing Cluster (一个GPU中有多个 Cluster)
* Streaming Multi-processor (SM, 一个 Cluster中有多个 SM)
  * share registers, 
  * 最细粒度的调度器
  * 任务给SM，SM来调度下面的小兵来执行这个程序，SM最小的调度单元是warp
* Warp [https://stackoverflow.com/questions/3606636/cuda-model-what-is-warp-size](https://stackoverflow.com/questions/3606636/cuda-model-what-is-warp-size)
  * minimal scheduling unit, SPs in same warp executes same instructions
  * 最小的调度单元（由多个 cuda core 构成）
  * 一个 warp 中的 cuda core 只能跑同一个程序，只是数据不同。
* SP（cuda core）,TESLA v100 一个 SM 上有 64 个 FP32 core，64个int32 core。。。
  * 32 SPs 构成一个 warp，32个 core 跑同一个程序，数据不同。



**CUDA 与 GPU**

* GPU 保证，**一个 block 中的线程会在同一个 SM 上执行**
  * 但是，一个 SM 上可以运行多个 blocks
* ​



**CUDA  Framework（API 从高层到底层）**

* CUDA Libraries （通用算法的计算）
  * cuBLAS
  * cuFFT
  * cuRAND
  * cuSPARSE
  * cuSOVER
  * NPP
  * cuDNN
  * Thrust
* CUDA runtime
  * 最通用的计算接口
* CUDA driver
  * context 和 module 管理
  * context : 一个 context 就是一个 GPU 上进程所需要的一些数据
  * module：



**线程框架**

> 写代码的时候不需要考虑线程放到哪个 SM 上，这个不用考虑

* thread （局部内存）
* block （shared memory）
* grid （Grid中也有共享的内存空间）



**内存类型**

* `__global__`
* `__shared__`
* `__device__`
* `__host__`



## 疑问

* 不同 kernel 间如何进行数据通信

# Cuda 中的线程层级

* 线程：
* 线程束：32个线程，GPU任何时候都不会执行低于 32 个线程。 （代码的最小执行单位）
* 线程块：线程束为执行单位的话，对于GPU来说还是太小了，所以：程序员以**线程块为单位** 启动核函数。（代码的启动单位）
 *  常用：要给线程块中由 32、64、128、256、512、1024个线程


* 线程束是最小执行单位，线程块是程序启动单位。线程束总是包含32个线程，这意味着，如果一个线程块包含256个线程，这256个线程并不会一起执行，GPU硬件会用8个线程束来执行这些线程。warp0， warp1，。。。warp7

CPU内存和GPU内存是通过PCIe进行数据传输的。两者进行数据传输还会经过L3

## 网格

在启动 kernel 的时候，可以指定是1-D、2-D、3-D网格启动
```c
gridDim.x;   // 以多少块启动的
gridDim.y;
gridDim.z;

blockIdx.x;  // 当前是第几块
blockIdx.y;
blockIdx.z;

blockDim.x; // 每个块启动的线程是多少
blockDim.y;
blockDim.z;

threadIdx.x; // 当前的线程号是多少
threadIdx.y;
threadIdx.z;
```

* 程序员应该将该任务的执行划分为 “大量的互不依赖的” 块
* 每个块和其它块在资源上应该互不依赖
* 大规模的并行执行仅在有大量独立块的条件下才能发挥作用
* 任何块之间的依赖性都会使程序“串行化” 执行


# CUDA编程

## 初始化 与 GPU信息查询

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

int num_gpus;
cudaGetDeviceCount(&num_gpus); //  获取本机有多少个gpu

cudaError_t cuda_status, cuda_status2;
cuda_status = cudaSetDevice(0);

cudaDeviceProp gpu_properties;
cudaGetDeviceProperties(&gpu_properties, 0);

unsigned long supported_k_blocks, supported_m_blocks, max_thr_per_block;
char supported_blocks[100];


gpu_properties.maxGridSize[0];
gpu_properties.maxGridSize[1];
gpu_properties.maxGridSize[2];
```


## PCIE 总线对性能的影响

内存、L3、GPU-L2，GPU内存

主板、CPU、GPU都必须支持某个PCIE速度，否则得到的PCIE速度就是三者最低的


* 数据传输时间：A到B传输所花费的时间。没有提及时间！！！！
* 速度：传输单位数据量所花费的时间
* 延迟：传输一批连续的数据包时，第一个数据包到达的时间
* 吞吐量：一段时间内，多个数据包的平均传输速度
* 带宽：支持的最大吞吐量
* PCIE：支持双向同时数据传输

## 全局内存总线对性能的影响


* CPU 与 dram DDR4：68GBps
* GPU 与 全局内存：336GBps

*  

# GPU 硬件架构

* SM：streaming processor，流处理器，有一个L1 cache，多个GPU core
* GPU core：都有 ALU、FPU
* 千兆线程调度器：将线程块分配给SM的 调度器，可以很快的完成块的分配。
 * 每个SM接收 线程块 的个数是由限制的，超过限制的话，SM就会阻止 调度器 向其分配线程块。所以 千兆线程调度器 有时会阻塞
* L1缓存：一个SM一个，同SM的GPU core共享
* L2缓存：全局一个，所有SM共享

```c++
// 设备端数组，不是主机端的，编译器将决定其去向！
__device__ double gauss[2][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
```

