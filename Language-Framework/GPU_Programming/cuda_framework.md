# 硬件结构与CUDA结构

**GPU Framework**

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
* SP（cuda core）
  * 32 SPs 构成一个 warp，32个 core 跑同一个程序，数据不同。





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