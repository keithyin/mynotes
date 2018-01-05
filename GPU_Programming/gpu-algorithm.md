# 基本 GPU 算法

* reduce
* scan



## reduce

* inputs :  **set of elements** , **reduction operators**
* outputs : **one elements**

```python
a = [1,2,3,4,5]
op = add
res = 15
```





## scan

* inputs :   **set of elements** , **scan operators**， **identity element**
  * I op element = element
* outputs : **set of elements** 

```python
a = [1,2,3,4,5]
op = add
res = [1,3,6,10,15] # inclusive
# or
res = [0,1,3,6,10] # exclusive
```

**特点：**

* 下一个输出 和 上一个输出 有关

|                 | More step efficient | more work efficient |
| --------------- | ------------------- | ------------------- |
| Hillis + Steele | no                  | Yes                 |
| Blelldch        | Yes                 | no                  |



## histogram

```c++
__global__ void naive_histo(int *d_bins, const int *d_in, const int BIN_COUNT){
	int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int myItem = d_in[myId];
	int myBin = myItem % BIN_COUNT;

	// read data from global storage and store it in local register.
	// d_bins[myBin] ++; // will cause race condition
	atomicAdd(&(d_bins[myBin]), 1);

}
```

* 上述算法由于 `atomicAdd` 会导致 速度变慢，不利于 扩展



**另一种方法**

* **per-thread private (local) histograms, then reduce**



128 items, 8 threads 3 bins.  items 多， bin 少

* 每个 thread 处理 16 个 items， 每个 thread 有 3 个 bins
* 然后 用 reduce 来 求 最终结果。



**另一种方法**

* **sort, then reduce by key**






