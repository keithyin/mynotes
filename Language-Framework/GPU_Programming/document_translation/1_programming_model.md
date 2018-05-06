# CUDA 编程（一）：programming model

**Programing Model 的三个核心抽象**

* 线程的层次结构 （线程 -> 线程块 -> 线程网格）（thread -> thread block -> thread grid）
* 共享 内存 (shared memory)
* barrier synchronization

这些抽象提供了**细粒度**的数据并行和线程并行，嵌入在**粗粒度**的数据并行和任务并行之中。这些抽象引导着编程人员将一个**大问题**分解成一些**粗粒度的子问题**，这些粗粒度的子问题可以被**并行**的由**线程块**解决，每个子问题又可以分解成**更细粒度**的问题，由线程块中的线程**合作**来解决。

总结来说：大问题 --> 子问题（每个子问题一个线程块，子问题之间独立）--> 子子问题（每个子子问题一个线程，子子问题之间可以协作）。



**以下将从四个方面来介绍CUDA 的 programing model**

* Kernel
* Thread Hierarchy （线程层次）
* Memory Hierarchy （内存层次）
* Heterogeneous Programming （混合编程）



## Kernel

我们只需要简单的在 `C` 函数前加个 `__global__` 修饰符就可以将一个 c 函数声明为 `kernel`，下面是一个例子

```c++
// Kernel definition 
__global__ void VecAdd(float* A, float* B, float* C) 
{ 
  int i = threadIdx.x; 
  C[i] = A[i] + B[i]; 
} 
int main() 
{ 
  ... 
  // Kernel invocation with N threads 
  VecAdd<<<1, N>>>(A, B, C); 
  ... 
}
```

在调用的时候使用 一种新的语法 `<<<...>>>` ([execution configuration](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)) 来调用 `kernel` 函数，如上面代码所示，在调用的时候会产生 `N` 个线程来执行 `kernel`。



## Thread Hierarchy

线程的层次为 ： **Thread --> Thread Block -->  Thread Grid**。

 i.e. 多个 **Thread** 构成 **Thread Block**，多个 **Thread Block** 构成 **Thread Grid**。

我们需要在调用 `kernel` 的时候指定：`Thread Block，Thread per Block` 的数量，可以通过 一维，二维，三维的方式指定，这里就涉及到了一个 类 `dim3`。 其中：`dim3(n,1,1) == dim3(n,1) == n`：

```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{ 
  int i = threadIdx.x; 
  int j = threadIdx.y; 
  C[i][j] = A[i][j] + B[i][j]; 
} 
int main() 
{ ... // Kernel invocation with one block of N * N * 1 threads 
  int numBlocks = 1; 
 dim3 threadsPerBlock(N, N); // 每个 block 中 thread 的数量
 MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 
 ... 
}
```

注意：**Thread Block** 中的 线程数量是有限制的，现代 GPU，每个 block 最大可以放 1024 个线程。



线程块之间要求是独立运行的，`GPU` 无法保证 线程块之间的运行先后顺序。但是在 线程块内，GPU 提供了一个同步机制 **barrier** （通过调用 `__syncthreads()` 实现）。

```c++
// Kernel definition 
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int j = blockIdx.y * blockDim.y + threadIdx.y; 
  // 线程块中的 所有线程 都执行到了这句，才可以往下执行，否则阻塞
  __syncthreads();
  if (i < N && j < N) C[i][j] = A[i][j] + B[i][j]; 
} 
int main() 
{ 
  ... 
  // Kernel invocation 
  dim3 threadsPerBlock(16, 16); 
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y); 
  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C); 
  ... 
}

```



## Memory Hierarchy

**GPU中的Memory 有以下几种类型**

* Global Memory  （所有线程都可访问的存储空间）（生命周期与应用程序一致）
* Shared Memory （线程块中的线程可以访问的共享存储空间）（生命周期与线程块一致）
* Local Memory （每个线程私有的存储空间）（生命周期与线程一致）

**还有两种只读的 Memory**

* Constant Memory Space （所有线程可访问）（生命周期与应用程序一致）
* Texture Memory Space （所有线程可访问）（生命周期与应用程序一致）



**还差如何分配这些内存**



## Heterogeneous Programming （混合编程）

在 CUDA 的世界里 称 主机(cpu+memory)为 **host**, 称 GPU 为 **device** 。

CUDA 编程模型认为 CUDA 线程运行在 GPU上，C代码运行在 host 上。

CUDA 编程模型还认为，host 和 device 的 memory spaces 是独立的。将他们分别称作 host memory 和 device memory。i.e.  device memory 对运行在 device 上的代码可见，host memory 对运行在 host 上的代码可见。

**host** 代码通过调用 CUDA runtime 来管理 GPU 的 `global，constant，texture memory space`。

![](../imgs/heterogeneous-programming.png)











