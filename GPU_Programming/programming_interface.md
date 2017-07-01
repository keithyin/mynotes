# 编程接口

> 用到了 CUDA 扩展的源文件需要用 `nvcc`编译

The **runtime** is introduced in [Compilation Workflow](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-workflow)。它提供了以下支持，支持 在`host` 上运行的 `C`代码：

* 分配和 解除分配 `device` 上的内存。
* 在 `host` 与 `device` 之间传输数据。
* 管理 多个设备构成的系统。
* ...

> 可以看到，由于有 `runtime` CUDA 设备上的资源 ，基本都可以使用 运行在 `host` 上的 `C` 代码来控制。
>
>  [Device Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory) gives an overview of the runtime functions used to manage device memory。



## CUDA C Runtime

**初始化**

> There is no explicit initialization function for the runtime; it initializes the first time a runtime function is called

初始化时，`runtime` 干了些啥：

* 对系统 中的 每个 设备 创建一个 `context`，这个 `context` 是个 `primary context` ，可以被 `host`线程共享。[CUDA Context](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)
* 作为创建 `context`  的一部分 ，`device code` 会 [just-in-time compiled](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)(if necessary)，然后加载到 `device` 内存中。

> When a host thread calls `cudaDeviceReset()`, this destroys the primary context of the device the host thread currently operates on (i.e., the current device as defined in [Device Selection](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)). The next runtime function call made by any host thread that has this device as current will create a new primary context for this device.



**Device Memory**

> the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. *Kernels operate out of device memory*, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory.

* Device memory can be allocated either as linear memory or as CUDA arrays.
* CUDA arrays are opaque(不透明)  memory layouts optimized for texture fetching. They are described in [Texture and Surface Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory).
* Linear memory exists on the device in a 40-bit address space, so separately allocated entities can reference one another via pointers, for example, in a binary tree.????
* CUDA Linear memory 
  * 分配：`cudaMalloc()`
  * 释放：`cudaFree()`
* 数据传输：
  * host->device ： `cudaMemcpy()`
  * device->host： `cudaMemcpy()`

```c++
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);// size 是 字节大小！！！

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A; //d_A是 host 上的一个指针，但是它指向device的内存空间。
    //下面用的 &d_A 是想要改变 d_A 的值，即d_A指向的地方。
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    // cudaMemcpy(to, from, ...)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 将计的结果从device 内存 拷贝到 host 内存上。
    // h_C contains the result in host memory。。官方示例代码，
    // h_C 在 host 上根本就没有分配呀，逗我呢。。。
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}
```

> Linear memory can also be allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in [Device Memory Accesses](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses), therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the cudaMemcpy2D() and cudaMemcpy3D() functions). The returned pitch (or stride) must be used to access array elements. The following code sample allocates a width x height 2D array of floating-point values and shows how to loop over the array elements in device code:

**使用cudaMallocPitch() 分配 2D array**

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
// 分配空间时：指针的指针，第二个pitch 指的是啥？
// width 为什么要这么指定，
// height 的话就是整数，很直觉。
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
      // 这边没有搞明白，为什么要 char* ，按字节寻址 pitch 就是行的大小？
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```



**使用cudaMalloc3D() 分配 3D array**

```c++
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```



> The reference manual lists all the various functions used to copy memory between linear memory allocated with cudaMalloc(), linear memory allocated with cudaMallocPitch() or cudaMalloc3D(), CUDA arrays, and memory allocated for variables declared in global or constant memory space.

**通过 runtime API 访问 global variables**

```c++
__constant__ float constData[256];
float data[256];
// cudaMemcpyToSymbol(to, from, size)
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

> cudaGetSymbolAddress() is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through cudaGetSymbolSize().
>
> 一般情况下，kernel 中的代码使用 blockIdx, threadIdx 来得知当前是哪个线程在执行它。



**共享内存：**

[变量类型修饰符，`__device__`...这些](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-type-qualifiers)

* `__device__` 来说明变量分配在设备上
* ​

> 使用 `__shared__` 变量修饰符来分配共享内存



* thread


* block
* grid



**page locked Host Memory**

> 这是给主机分配内存的 runtime API 哦

* 分配：`cudaHostAlloc()`
* 释放：`cudaFreeHost()`
* `cudaHostRegister()` 用来注册 `malloc()` 分配的内存作为 `page locked` 的。

Using page-locked host memory has several benefits:

- Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices as mentioned in [Asynchronous Concurrent Execution](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution).
- On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory as detailed in [Mapped Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory).
- On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in [Write-Combining Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#write-combining-memory).





## 参考资料

[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

