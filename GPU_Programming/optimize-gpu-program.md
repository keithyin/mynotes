# 如何优化 GPU 代码



* latency (`time, seconds`) (CPU 优化 latency) 干成一件事所需要的时间
* throughput  (`stuff/time` , `jobs/hour`) (GPU focus on throughput) 单位时间内做了多少事
  * 也叫 `band-width`
* `TFLOPS` : trillion floating point operation per second.   3TFLOPS




## 如何分析一个算法

* 并行化做的怎么样，是不是有大量 idle 线程
* 访问内存的行为 怎么样， 需不需要用 `shared memory`
* thread divergence 如何？？
* 算法的复杂度如何？ step complexity， work complexity
  * 希望和 serial algorithm 有相同的 work complexity，但是更少的 step complexity
  * 如果 work complexity 多，step complexity 少也是可以。





## High Level Strategies

**maximize arithmetic intensity**
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
  * 为什么 coalesce 好，如果 线程的调度不是 相邻的线程一起跑的呢？实际上是 一个 warp 一起 搞的，所以 coalesce 有用。




**Avoid thread divergence**

> 不同的 thread 做了不同的事情， 例如 if-else 分支(有些走 if 分支，有些 走 else 分支)， loop(不同的 thread 可能循环的次数不同。



**减少原子操作 和 同步操作**



**减少 访问 global memory 的次数：使用 shared memory**



**将一个问题 划分成多个 独立的子问题，使其并行化**

**减少 线程间的通信**

- 这里有和 线程数 有一个 trade-off,  线程间的通信 慢于 线程内  local memory 的操作。

> 进程间的通信： 得到最终的结果 需要其它线程的结果。



## 使用 shared memory

**什么情况下使用 shared memory**

* 因为 shared memory 是 对于 block 而言的，所以通过 block 来分析比较靠谱
* 如果 `访问global memory次数 per block` ，大于 block 中的线程数，    就可以考虑使用 `shared memory`








## Demos

**N-Body**

> n 个元素，每个元素对其它元素都有力， 计算每个元素身上受的合力

**如何加速呢？**

* 减少 访 global memory 次数： 使用 `shared memory`
* ​

**spMV**

* keep your threads busy
* manage communication is important. 
  * communicating through registers is faster than communicating through shared memory.



