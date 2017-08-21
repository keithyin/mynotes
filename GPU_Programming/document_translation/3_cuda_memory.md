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

我们一般使用 `cudaMalloc()` 来分配 `linear memory`， 使用 `cudaFree()` 释放 `linear memory`。

```c++
// Device code 
__global__ void VecAdd(float* A, float* B, float* C, int N) 
{ 
  int i = blockDim.x * blockIdx.x + threadIdx.x; 
  if (i < N) C[i] = A[i] + B[i]; 
} 
// Host code 
int main() 
{ 
  int N = ...; 
  size_t size = N * sizeof(float); 
  // Allocate input vectors h_A and h_B in host memory 
  float* h_A = (float*)malloc(size); 
  float* h_B = (float*)malloc(size); 
  // Initialize input vectors 
  ... 
  // Allocate vectors in device memory 
  float* d_A; 
  cudaMalloc(&d_A, size); 
  float* d_B; 
  cudaMalloc(&d_B, size); 
  float* d_C; 
  cudaMalloc(&d_C, size); 
  // Copy vectors from host memory to device memory 
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
  // Invoke kernel 
  int threadsPerBlock = 256; 
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); 
  // Copy result from device memory to host memory 
  // h_C contains the result in host memory 
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); 
  // Free device memory 
  cudaFree(d_A); 
  cudaFree(d_B); 
  cudaFree(d_C); 
  // Free host memory 
  ... 
}

```

下面代码展示了 使用 runtime 访问 global variable 的一些其它方法：

```c
__constant__ float constData[256]; 
float data[256]; 
cudaMemcpyToSymbol(constData, data, sizeof(data)); 
cudaMemcpyFromSymbol(data, constData, sizeof(data)); 
__device__ float devData; 
float value = 3.14f; 
cudaMemcpyToSymbol(devData, &value, sizeof(float)); 
__device__ float* devPointer; float* ptr; cudaMalloc(&ptr, 256 * sizeof(float)); 
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```



## shared memory

> 线程块内线程的共享内存

分配共享内存有两种方法：

```c++
// 在 kernel 代码中分配
__global__ void cube(float * d_out, float * d_in, int value)
{
  	//这种情况下，__shared__ float mem[value] value 不能是变量，必须在编译前指定好
	__shared__ float mem[100];
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f*f;
}
```

```c
// 使用 extern 的方法分配，这种方法是可以在调用的时候动态指定分配多少空间的。
extern __shared__ float array[]; //这里不用指定分配多少 空间，在 调用kernel的时候指定
__global__ void func() // __device__ or __global__ function 
{ 
  short* array0 = (short*)array; 
  float* array1 = (float*)&array0[128]; 
  int* array2 = (int*)&array1[64]; 
}

int main()
{
  int size_of_shared_mem = 1000;
  func<<<1,1, size_of_shared_mem*sizeof(float)>>>();
}
```

[参考资料 https://stackoverflow.com/questions/28832214/why-should-i-use-cuda-shared-memory-as-extern](https://stackoverflow.com/questions/28832214/why-should-i-use-cuda-shared-memory-as-extern)



## page-locked host memory

这个和前面讨论的略有不同，前面都是介绍分配 `device` 内存的，这个是介绍 分配 `host` 内存的。

`cuda runtime` 提供了一些函数允许使用 `page-locked（pinned）` host memory（传统的 host memory 是用 malloc 分配的）。

分配与释放：

* `cudaHostAlloc()` 分配 pinned 内存
* `cudaFreeHost()` 释放 分配的pinned 内存
* `cudaHostRegister()` 用来 page-lock 使用 `malloc` 分配的 memory.



使用 page-locked host memory 有以下优点（仅对某些设备而言，并不是所有 n 卡都适用）：

* **device 和 host 之间的数据传输** 与 **kernel 的执行**可以并行起来执行。
* pinned memory 可以映射到 device 的地址空间，这样就不需要 device 与 host 之间的数据传输了。 
* 在拥有 front-side bus 的系统中，device memory 和 pinned memory 之间的带宽是很高的。

`page-locked memory` 是个稀缺资源，分配过多的 `page-locked memory` 会使得操作系统的性能下降，因为可供调页的空间不多了。



**Portable Memory**

一个 `flag` ， `cudaHostAllocPortable，cudaHostRegisterPortable` ,可以在调用`cudaHostAlloc()，cudaHostRegister()`传入，暂时没稿清楚有啥用。

**Write-Combining Memory**

一个 `flag` ，`cudaHostAllocWriteCombined` ，也是传入那两个函数中的。

**Mapped Memory**

一个 `flag` ，`cudaHostAllocMapped，cudaHostRegisterMapped`，也是传入那两个函数中的。



