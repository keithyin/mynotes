# 如何优化 GPU 代码



* latency (`time, seconds`) (CPU 优化 latency) 干成一件事所需要的时间
* throughput  (`stuff/time` , `jobs/hour`) (GPU focus on throughput) 单位时间内做了多少事
  * 也叫 `band-width`
* `TFLOPS` : trillion floating point operation per second.   3TFLOPS




## High Level Strategies

* maximize arithmetic intensity
  * 访问了一次 内存， 做 更多的 操作。
  * maximize work **(useful compute operations)**  per thread
  * minimize **time spent on** memory per thread

$$
arithmetic\_internsity = \frac{math}{memory}
$$

**如何优化 花费在 读取内存上的时间**

* 将经常访问的 数据 放在 快速内存上(`global memory ---> shared memory--> local memory`)
* **coalesce global memory access**
  * 当 GPU read/write global memory 的时候，它会一次性 **访问 一大块  存储空间**，
  * 所以当 相邻的线程 read/write 相邻的 global memory 的话，速度就会很快。
  * 疑问？ 访问一大块存储空间是什么意思，放到 cache 里面吗？ 




**Avoid thread divergence**

> 不同的 thread 做了不同的事情， 例如 if-else 分支(有些走 if 分支，有些 走 else 分支)， loop(不同的 thread 可能循环的次数不同。)