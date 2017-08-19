# CUDA 编程（一）：programing model

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











