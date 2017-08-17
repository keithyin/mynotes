# CUDA declaration specifier 总结



## 修饰 函数的修饰符

* `__global__` 修饰符：用来声明一个函数是 ` kernel`, `kernel` 具有以下特征
  * 在 `host` 上调用 (callable from the host)
  * 在 `device` 上执行

```c++
__global__ void cuda_kernel(int a, int b){
  // do something on gpu
}
```

* `__device__` 修饰符： 用来声明一个 函数，具有以下特征
  * 只能在 `device` 上调用
  * 在 `device` 上执行
* `__host__` 修饰符：用来声名一个函数，具有以下特征
  * 只能由 `host` 调用
  * 在 `host` 上执行




**`__global__ vs __device__`**

* 被 `__global__` 修饰的函数 称之为 `kernel`


* `__device__` 修饰的函数只能由  `__device__` 修饰的函数或 `__global__` 修饰的函数调用！
* `__global__` 修饰的函数可以 由 `host` 调用，通过 `<<<...>>>`
* `__device__` 修饰的函数调用的时候不需要 `<<<...>>>`
* `__device__` 修饰的函数，在  `gpu` 线程上执行。

参考资料：

[https://code.google.com/archive/p/stanford-cs193g-sp2010/wikis/TutorialDeviceFunctions.wiki](https://code.google.com/archive/p/stanford-cs193g-sp2010/wikis/TutorialDeviceFunctions.wiki)

[https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions](https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions)









## 修饰变量的修饰符

> 变量修饰符 指定了 变量在 设备上的 memory location

* `__device__` 修饰符：声明变量 驻留在 `device` 上，修饰的变量具有以下特征

  * 驻留在 `global memory` 上
  * 和应用程序有相同的生命周期
  * `grid` 中的所有线程都可以访问 （通过 runtime library）(`cudaGetSymbolAddress(),cudaGetSymbolSize(), cudaMemcpyToSymbol(), cudaMemcpyFromSymbol()`)

* `__constant__` 修饰符：可以和 `__device__` 一起使用，修饰的变量具有以下特征

  * 驻留在 内存常量区
  * 和应用程序具有相同的生命周期
  * `grid` 中的所有线程都可以访问（通过 runtime library） (`cudaGetSymbolAddress(),cudaGetSymbolSize(), cudaMemcpyToSymbol(), cudaMemcpyFromSymbol()`)

* `__shared__` 修饰符：可以和 `__device__` 一起使用，声明的变量具有以下特征

  * 驻留在线程块的 共享内存区
  * 和线程块具有相同的 生命周期
  * 只有线程块中的线程可以访问




```c++
// 可以用这种方式来 分配 shared memory
extern __shared__ float array[]; // array 的大小是运行时通过 Execution Configuration
// 这句 可以写在 kernel 里也可以写在 kernel 外
// 但是用了这句之后，一定要用 Execution Configuration
__device__ void func()
{
  short* array0 = (short*) array;
  float* array1 = (float*) &array0[128];
  int* array2 = (int*) &array1[64];
}
```



## Execution Configuration

在调用 `__global__` 函数时，必须为这次调用指定 `Execution Configuration`，就是 这玩意

 `<<<gridDim, blockDim, Ns, S>>>`

* `gridDim` : `dim3` 类型，用来指定 grid 的 size和 维度
* `blockDim`： `dim3` 类型，用来指定 block 的 size 和 维度
* `Ns`：`size_t` 类型，指定 每个 block 的 shared memory 的 大小（number of bytes）可选的，默认为0
* `S`：`cudaStream_t` 类型，指定相关联的 `stream`，可选，默认为0





## 参考资料

[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)

[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)

[https://code.google.com/archive/p/stanford-cs193g-sp2010/wikis](https://code.google.com/archive/p/stanford-cs193g-sp2010/wikis)

[https://stackoverflow.com/questions/33218522/cuda-host-device-variables](https://stackoverflow.com/questions/33218522/cuda-host-device-variables)

[http://www.math-cs.gordon.edu/courses/cps343/presentations/CUDA_Memory.pdf](http://www.math-cs.gordon.edu/courses/cps343/presentations/CUDA_Memory.pdf)





