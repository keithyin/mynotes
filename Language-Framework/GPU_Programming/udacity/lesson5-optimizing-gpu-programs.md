# Optimizing GPU Programs

**想要编写一个高效的GPU程序需要注意的几件事**

* **coalesce global memory accesses, **
  * 读写都要注意内存的连续
  * 因为每次访存都是以 chunk 的形式来的。
* **avoid thread divergence, **
  * (减少线程中的跳转操作。)
* **decrease time spent on memory operations**
  * 降低在内存访问上所消耗的时间
    * 可以将经常访问的数据移动到 shared-memory 上来实现



**arithmetic intensity**
$$
 = \frac{math}{memory}
$$

* math per memory， 每个 memory 读出来后，计算了多少次
* 越大越好



**优化的级别**

* 选择一个好的算法 
  * 选择一个 fundamentally parallel 的算法
* 遵循基本的 GPU 编程原则
  * coalescing global memory
  * use shared memory
* 架构级别上的优化
  * bank conflicts
  * optimizing register
* $\mu$-optimization at instruction level 
  * floating point
  * denorm hacks



![](../../imgs/apod-1.png)

**代码优化流程：APOD**

* **Analyze**: Profile whole application 
  * where can it benefit
  * by how much?
* **Parallelize: ** 
  * Pick an approach
    * 找个好的 libraries
    * Programming Language
  * Pick an algorithm
    * 这个非常重要
* **Optimize**
  * profile-driven optimization
  * Don't optimize in a vacuum
  * 在真实环境中测试！！！！
* **Deploy**