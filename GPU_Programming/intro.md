# intro

> The challenge is to develop application software that transparently scales its parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their parallelism to manycore GPUs with widely varying numbers of cores.
>
> The CUDA parallel programming model is designed to overcome this challenge while maintaining a low learning curve for programmers familiar with standard programming languages such as C.



**三个核心抽象：**

* a hierarchy of thread groups
* shared memories
* barrier synchronization

> They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.



## 编程模型

**Kernels**

> 在 `cuda` 中，在定义 `kernel`的时候需要使用 `__global__` 来声明。使用 `<<<...>>>` 配置语法来指定 用多少个线程来执行来执行 `kernel`。执行 `kernel` 的线程都有一个唯一的 `id`，在 `kernel` 中可以通过 `threadIdx` 来读取。



```c++
// kernel 定义
__global__ void VecAdd(float* A, float* B, float* C){
  int i = threadIdx.x;
  C[i] = A[i]+B[i]
}
int main(){
  //用 N 个线程来调用 kernel
  VecAdd<<<1,N>>>(A,B,C);
}
```



**线程层次：**

> threadIdx 是 三个元素的 vector。所以，线程可以通过 一维的，二维的，或三维的线程索引形成一个 一维的，二维的，或三维的线程块（thread block）。这就为计算 向量啦，矩阵啦，`valume` 啦提供一个很自然的计算方法。
>
> The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size *(Dx, Dy)*,the thread ID of a thread of index *(x, y)* is *(x + y Dx)*; for a three-dimensional block of size *(Dx, Dy, Dz)*, the thread ID of a thread of index *(x, y, z)* is *(x + y Dx + z Dx Dy)*.



```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```



> 注意： 每个 block 中的线程数量是有限的，由于 一个block 中的所有线程 期望 驻留在*同一个* `processor core` 上，而且一定要共享 那块核心上的资源。当前GPU，一个 `core` 上的线程数最多不超过 1024 个。
>
> 但是，`kernel` 可以运行在多个 相同形状的 线程块上（thread blocks）。

```c++
// 在多个 blocks 上运行 MatAdd
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // Kernel invocation
 	// 可以看出 <<<...>>> 里面的参数可以是 int，也可以是 dim3 对象   
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```



> `Thread Blocks` 之间 需要是运行独立的，各 `block` 的运行不能有任何依赖关系。
>
> `block` 内的线程是可以协作运行的，通过共享内存。我们可以通过在 `kernel` 中调用 `__syncthreads()` 指定 同步点。 `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed. [Shared Memory](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) gives an example of using shared memory.
>
> For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and __syncthreads() is expected to be lightweight.



**内存层次：**

> CUDA 的内存层次有： 线程局部内存，block 共享内存(对block内的所有线程可见)，全局内存（所有线程共享）。
>
> 还有两个所有线程都可读的内存：constant and texture memory spaces。



**混合编程：**

> CUDA 编程模型 *假设* CUDA线程运行在物理上独立的设备上，这个设备与运行 C 程序的主机构成协同处理的关系。
>
> 意思就是：当 `kernel`运行在 `GPU` 上的时候，其它的 `C` 代码在 `CPU`上执行。
>
> CUDA 编程模型同时也 *假设* ，`host` 和 `device` 拥有它们独立的内存空间。所以这就涉及到了，`device` 上存储空间的分配和回收，和 `host`与`device`之间的数据传输。
>
> Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in [Programming Interface](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)). This includes device memory allocation and deallocation as well as data transfer between host and device memory.
>
> Note: Serial code executes on the host while parallel code executes on the device.

![](./imgs/heterogeneous-programming.png)

