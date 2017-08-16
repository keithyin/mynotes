# 一段代码搞懂 gpu memory

**GPU 的 memory 分为 三种，io速度从快到慢排序为：**

* local memory
* shared memory
* global memory

其中 shared memory 的io 速度是远快于 global memory 的。



**这三种 memory 的访问性质是：**

* local memory: 线程私有，只能本线程访问
* shared memory: 线程块(thread block) 共享, **同一个线程块**中的线程可以访问。
* global memory: 所有线程都可访问。



那么在编程的过程中，这三种 memory 是从什么地方体现出来的呢？



```c
#include <stdio.h>

__global__ void memory_demo(float* array)
{
	// array 指针是在 local memory 上的，但是它指向的 memory 是 global memory
	// i, index 都是 local variable，每个 线程 私有。
	int i, index = threadIdx.x;

	// __shared__ variable 对 block 中的 线程可见
	// 并 和 thread block 有相同的 生命周期。
	__shared__ float sh_arr[128];

	// 将 global memory 的值 拷贝到 shared memory 上。
	sh_arr[index] = array[index];

	// barrier here
	__syncthreads();
}

int main()
{
	float h_arr[128];
	float *d_arr;

	// cudaMalloc 分配的 memory 是在 global memory 上的。
	cudaMalloc((void **)&d_arr, sizeof(float)*128);
	cudaMemcpy((void*) d_arr, (void*) h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);

	// 启动 kernel
	memory_demo<<<1, 128>>>(d_arr);

	// .. do other stuff
}
```

