#include <stdio.h>

__global__ void memory_demo(float* array)
{
	// the point array is in local memory and it points to global memory
	// local variable, private to each thread
	int i, index = threadIdx.x;

	// __shared__ variable are visiable to all threads in the thread block
	// and have the same lifetime as the thread block
	__shared__ float sh_arr[128];

	// copy data from 'array' in global to sh_arr in shared memory
	sh_arr[index] = array[index];

	// barrier here
	__syncthreads();
}

int main()
{
	float h_arr[128];
	float *d_arr;

	// allocate gloabl memory on the device, place result in d_arr
	cudaMalloc((void **)&d_arr, sizeof(float)*128);
	cudaMemcpy((void*) d_arr, (void*) h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);

	// launch the kernel
	memory_demo<<<1, 128>>>(d_arr);

	// .. do other stuff
}