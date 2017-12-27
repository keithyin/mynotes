# 5：多设备系统

**一个 host system 可以有多个 devices** 



## 枚举系统上的设备

```c++
#include <cstdio>

int main(){
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int device=0; device<deviceCount; ++device){
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
	}
	return 0;
}
// Device 0 has compute capability 5.0.
```



## 选择设备

[http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior)

通过调用 `cudaSetDevice` , `host thread` 可以在任何时候设置 其要 操作的 `device`。

* 设备 `memory` 分配 和 `kernel launches` 都是执行在 当前设备上的。
* `streams` 和 `events` 也是 与 当前 `device` 相关的。

```c++
size_t size = 1024 * sizeof(float); 
// 设置当前 device 上下文。
cudaSetDevice(0); // Set device 0 as current 
float* p0; 
cudaMalloc(&p0, size); // Allocate memory on device 0 
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0 
cudaSetDevice(1); // Set device 1 as current 
float* p1; 
cudaMalloc(&p1, size); // Allocate memory on device 1 
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```



```c++
cudaSetDevice(0); // Set device 0 as current 
cudaStream_t s0; 
cudaStreamCreate(&s0); // Create stream s0 on device 0 
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0 
cudaSetDevice(1); // Set device 1 as current 
cudaStream_t s1; 
cudaStreamCreate(&s1); // Create stream s1 on device 1 
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1 

// This kernel launch will fail: stream 是和 device 相关的。
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0
```



**A memory copy will succeed even if it is issued to a stream that is not associated to the current device.**



## 设备间 Memory Access

**a kernel executing on one device can dereference a pointer to the memory of the other device**



需满足条件

* 64-bit
* compute capability >=2.0
* `cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)`  返回 True。



```c++
cudaSetDevice(0); // Set device 0 as current 
float* p0; 
size_t size = 1024 * sizeof(float); 
cudaMalloc(&p0, size); // Allocate memory on device 0 
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0 
cudaSetDevice(1); // Set device 1 as current

cudaDeviceEnablePeerAccess(0, 0); // Enable peer-to-peer access with device 0 

// Launch kernel on device 1 
// This kernel launch can access memory on device 0 at address p0 
MyKernel<<<1000, 128>>>(p0);

```



```c++
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
```



```c++
cudaSetDevice(0); // Set device 0 as current 
float* p0; 
size_t size = 1024 * sizeof(float); 
cudaMalloc(&p0, size); // Allocate memory on device 0 
cudaSetDevice(1); // Set device 1 as current 
float* p1; 
cudaMalloc(&p1, size); // Allocate memory on device 1 
cudaSetDevice(0); // Set device 0 as current 
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0 
cudaSetDevice(1); // Set device 1 as current 
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1 
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```





