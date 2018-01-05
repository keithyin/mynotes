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





## scan  !!!!!

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




## compact

>  given a set, we want to filter that set to get a subset that we care about.

也可以叫做: filter, roi



* inputs : set of elements, predicate
* outputs : compacted result



```python
in_ = [s0, s1, s2, s3, s4, ...]
predicate = [True, False, True, False, True, ...]
output = [s0, Null, s2, , Null, s4, ...] # sparse
# or
output = [s0, s2, s4, ...] # dense， !!!better!!! that is what we want!!!
```

* 得到 dense 的结果对后续的计算是非常好的。不会有很多线程 idle



**当 有用的元素很少 或 有用元素计算量很大， 用 compact**





**核心算法**

```python
predicate = [True, False, False, True, True, False, True, False]
addresses = [0, -, -, 1, 2, -, 3, -]

predicate = [1, 0, 0, 1, 1, 0, 1, 0] 
addresses = [0, 1, 1, 1, 2, 3, 3, 4] # do scan operation on the predicate
```

**compact 的步骤**

* 计算 predicate
* 将 predicate 转成 [0,1] 序列
* 对 [0,1] 序列 执行 exclusive-sum-scan operation， 得到 addresses
* scatter input into output using addresses



**理解 compact**

* 对 true inputs 分配 **1** 个 item
* 对 false inputs 分配 **0** 个 item

**泛化 compact**

* **the number of output items** can be computed dynamically for each input item



## Segmented Scan

