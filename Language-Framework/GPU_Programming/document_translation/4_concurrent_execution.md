# CUDA编程（四）：Concurrent Execution

## [Asynchronous Concurrent Execution](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)

**CUDA 将以下操作看作独立的 task，他们之间可以并行执行：**

- computation on the host
- computation on the device
- Memory transfer from the host to the device
- Memory transfer from the device to the host
- Memory transfers within the memory of a given device
- Memory transfers among devices




**host 和 device 之间的并行执行**

在请求 device 的任务完成之前，CUDA 就会将控制权返回给 host 线程。使用这种异步的调用，host 的请求可以排成队列等待 GPU 的调度。这也将 host 从管理设备device的工作中解脱出来，使得它可以干一些其它的事情。下列的 **device  operation** 对于 **host** 来说是异步的：

- Kernel Launches
- `cudaMemcpy`  within the same device
- Memory copies from **host to device** of a memory block of 64 KB or less;
- `cudaMemcpy*Async`
- `cudaMemset*Async` 

**注意：如果 host memory 不是 page-locked，即使指定了 Async 的数据传输也是 Sync 的。**



我们可以通过设置 环境变量 `CUDA_LAUNCH_BLOCKING=1` 来禁止这种异步特性。这种特性仅是为了 debug 存在的。



**Concurrent Kernel Execution（多个 CUDA kernel 共同在 GPU 上计算）**

计算能力 > 2.x 的卡可以**多Kernel** 并行计算。



**Overlap Data Transfer and Kernel Execution（Data Transfer operation 与 Kernel 计算并行执行）**

一些设备支持 **host-device 之间的数据传输** 与 **kernel 的执行**并行执行。



**Concurrent Data Transfer（多个 Data Transfer operation 可以一起执行）**

计算能力 > 2.x 的卡可以**多Kernel** 可以 **overlap copies to and from device** 



**想要 CDUA并行 特性，所需要的条件**

* CUDA operations 必须在 **不同** 的 stream 上（同一个 stream 上的 operation 是串行的）
* `cudaMemcpyAsync` 操作的 host  memory 必须是 `pinned` 
* 必须有足够的资源才能支撑起 并行（registers，。。。）



## [Streams](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)

应用程序通过 **stream** 来管理 **concurrent operations**。什么是 stream 呢？`stream`：**A sequence of operations that execute in issue-order on the GPU**



* CUDA operations **in different streams** may run concurrently
* CUDA operations **from different streams** may be interleaved





**如何创建和销毁一个 stream**

创建一个 stream 对象，然后将它作为参数传给一系列的 `kernel launches` 和 `host--device memory copies` 函数，这就创建好了一个 stream了。

```c++
cudaStream_t stream[2]; 
for (int i = 0; i < 2; ++i) 
  cudaStreamCreate(&stream[i]); 
float* hostPtr; 
cudaMallocHost(&hostPtr, 2 * size);

for (int i = 0; i < 2; ++i) 
{ 
  cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]); 
  MyKernel <<<100, 512, 0, stream[i]>>> (outputDevPtr + i * size, inputDevPtr + i * size, size); 
  cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]); 
}
```

通过调用 `cudaStreamDestroy()` 来销毁 stream

```c
for (int i = 0; i < 2; ++i) 
  cudaStreamDestroy(stream[i]);
```



**default stream**

如果调用 `kernel` 和 调用`runtime` 的时候没有指定 `stream`， 会有一个默认的 `stream` 被指定。 



**Explicit Synchronization（显式同步的方法）**

CUDA 提供了很多方法显示的同步 stream 。

* `cudaDeviceSynchronize()`
  * Block **host** until all issued CUDA calls are complete

- `cudaStreamSynchronize(streamid)` , 这个函数将 `stream` 作为参数，
  - Block **host** until all CUDA calls in stream id are complete.
- 使用 `Events` 进行同步
  - 在 stream 中创建一个 特定的 `Event`
  - `cudaEventRecord(event, streamid)`
  - `cudaEventSynchronize(event)`
  - `cudaStreamWaitEvent(stream, event)`
  - `cudaEventQuery(event)`



```c++

cudaEvent_t event;                                // create event
cudaEventCreate (&event); 
cudaMemcpyAsync ( d_in, in, size, H2D, stream1 ); // 1) H2D copy of new input
cudaEventRecord (event, stream1);                 // record event

cudaMemcpyAsync ( out, d_out, size, D2H, stream2 ); // 2) D2H copy of previous result
cudaStreamWaitEvent ( stream2, event );             // wait for event in stream1
kernel <<< , , , stream2 >>> ( d_in, d_out );       // 3) must wait for 1 and 2
asynchronousCPUmethod ( ... ) // Async GPU method
 
```



**Implicit Synchronization（隐式同步）**

下列的 operations 会隐式的同步 所有其它 CUDA operations,:

* page-locked memory allocation
  * `cudaMallocHost`
  * `cudaHostAlloc`
* device memory allocation
  * `cudaMalloc`
* Non-Async version of memory operations
  * `cudaMemcpy*`
  * `cudaMemset*`
* change to L1/shared memory configuration
  * `cudaDeviceSetCacheConfig`

