# CUDA编程（二）：Programming Interface（一）

>  **本文主要是对 官方文档中的 [ Compilation with NVCC](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc) 部分进行总结。**



如果对 C 编程很熟悉的话，使用 CUDA C 写 CUDA 代码是非常容易的。

**CUDA C 包含了**

* 对 C 语言的一些扩展 （`<<<...>>>` 和一些 修饰符 `__global__, __device__` 等等。。。）
* 一个 runtime library

任何包含了 以上扩展的源代码都必须要要用 **nvcc** 编译。

```shell
nvcc -o func func.cu
```



**runtime library:**

提供了一些**在 host 上执行的C 函数**，可以用来：分配和回收设备内存，host 和 device 间的数据传输，管理多设备系统 等等。



`nvcc` 可以编译 混合着  `host code（运行在 host 上的代码）` 和  `device code（运行在 device 上的代码）` 的源文件，`nvcc` 的 基本工作流程是：

* 将 `device code` 从 `host code` 中剥离开来
* 将 `device code` 编译成二进制文件（cubin object）
* 修改 `host code` ：将 `<<<...>>>` 替换成必要的 `runtime function`，这些 `function` 是用来加载和运行编译好的 `kernel` 代码



被修改的 `host code` 既可以以 c 代码的形式输出（之后可以用其它的工具来编译之），也可以直接由 `nvcc` 在编译的最后阶段调用 `host` 编译器来编译。





