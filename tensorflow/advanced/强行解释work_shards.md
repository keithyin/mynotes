# 强行解释 work_shard

> 在学习 tensorflow 自定义 op 的时候碰到的，google 了一下，也没有找到详细的介绍，难道是姿势不对？？ 
>
> 通过看 了一些示例，这里打算强行解释一波。



## 概览

**如果想用 work shard，首先 代码能够并行化计算。work shard 是一个代码并行化工具。不用自己头疼的写多线程代码了。**

**什么样的代码能够并行化计算 ---> 每一个输出数据都能表示成相互无关的**



**work_shard 的最后一个参数就是要 shard 的 work， 这个 work 的签名为 void shard(int64 start, int64 limit)，work_shard 就是将 (start, limit) 给划分成多块，然后 块给 一个线程来计算。**



``` python
# 如何使用 work_shard
# 1. 包含头文件
# 2. 该用的地方用就行了
# 3. 链接的时候 g++ 会自动找到实现去链接的，不用操心。
```





## 代码

**work_shard声明代码** [地址](https://github.com/KeithYin/Faster-RCNN_TFpy2/blob/master/lib/roi_pooling_layer/work_sharder.h)

```c++
// work_sharder.h
#ifndef TENSORFLOW_UTIL_WORK_SHARDER_H_
#define TENSORFLOW_UTIL_WORK_SHARDER_H_

#include <functional>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Shards the "total" unit of work assuming each unit of work having
// roughly "cost_per_unit". Each unit of work is indexed 0, 1, ...,
// total - 1. Each shard contains 1 or more units of work and the
// total cost of each shard is roughly the same. The calling thread and the
// "workers" are used to compute each shard (calling work(start,
// limit). A common configuration is that "workers" is a thread pool
// with at least "max_parallelism" threads.
//
// "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
// if not CPU-bound) to complete a unit of work. Overestimating creates too
// many shards and CPU time will be dominated by per-shard overhead, such as
// Context creation. Underestimating may not fully make use of the specified
// parallelism.
//
// "work" should be a callable taking (int64, int64) arguments.
// work(start, limit) computes the work units from [start,
// limit), i.e., [start, limit) is a shard.
//
// REQUIRES: max_parallelism >= 0
// REQUIRES: workers != nullptr
// REQUIRES: total >= 0
// REQUIRES: cost_per_unit >= 0
void Shard(int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_WORK_SHARDER_H_
```

**用到 Sharder 的地方（见代码片段最后）** [完整代码地址](https://github.com/KeithYin/Faster-RCNN_TFpy2/blob/master/lib/roi_pooling_layer/roi_pooling_op.cc)

```c++
auto shard = [pooled_height, pooled_width, spatial_scale,
num_rois, batch_size, data_height, data_width, num_channels,
&bottom_data_flat, &bottom_rois_flat, &output, &argmax]
(int64 start, int64 limit) {
for (int64 b = start; b < limit; ++b)
{
  // (n, ph, pw, c) is an element in the pooled output
  int n = b;
  int c = n % num_channels;
  n /= num_channels;
  int pw = n % pooled_width;
  n /= pooled_width;
  int ph = n % pooled_height;
  n /= pooled_height;

  const float* bottom_rois = bottom_rois_flat.data() + n * 5;
  int roi_batch_ind = bottom_rois[0];
  int roi_start_w = round(bottom_rois[1] * spatial_scale);
  int roi_start_h = round(bottom_rois[2] * spatial_scale);
  int roi_end_w = round(bottom_rois[3] * spatial_scale);
  int roi_end_h = round(bottom_rois[4] * spatial_scale);

  // Force malformed ROIs to be 1x1
  int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
  int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
  const T bin_size_h = static_cast<T>(roi_height)
  / static_cast<T>(pooled_height);
  const T bin_size_w = static_cast<T>(roi_width)
  / static_cast<T>(pooled_width);

  int hstart = static_cast<int>(floor(ph * bin_size_h));
  int wstart = static_cast<int>(floor(pw * bin_size_w));
  int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
  int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));

  // Add roi offsets and clip to input boundaries
  hstart = std::min(std::max(hstart + roi_start_h, 0), data_height);
  hend = std::min(std::max(hend + roi_start_h, 0), data_height);
  wstart = std::min(std::max(wstart + roi_start_w, 0), data_width);
  wend = std::min(std::max(wend + roi_start_w, 0), data_width);
  bool is_empty = (hend <= hstart) || (wend <= wstart);

  // Define an empty pooling region to be zero
  float maxval = is_empty ? 0 : -FLT_MAX;
  // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
  int maxidx = -1;
  const float* bottom_data = bottom_data_flat.data() + roi_batch_ind * num_channels * data_height * data_width;
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      int bottom_index = (h * data_width + w) * num_channels + c;
      if (bottom_data[bottom_index] > maxval) {
      maxval = bottom_data[bottom_index];
      maxidx = bottom_index;
      }
  	}
  }
  output(b) = maxval;
  argmax(b) = maxidx;
  }
};

const DeviceBase::CpuWorkerThreads& worker_threads =
*(context->device()->tensorflow_cpu_worker_threads());
const int64 shard_cost =
num_rois * num_channels * pooled_height * pooled_width * spatial_scale;

// 用到 shard 的地方
Shard(worker_threads.num_threads, worker_threads.workers, output.size(), shard_cost, shard);
```



**通过调用方法来分析Shard 声明中各参数的意义：**

```c++
// 声明
void Shard(int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work);

// 调用
Shard(worker_threads.num_threads, worker_threads.workers, output.size(), shard_cost, shard);

// max_parallelism: 最大并行个数，通过调用的形式来看，一般是使用 本机的线程数。
// workers: 从声明来看，是代表的线程池。
// total: 从调用来看，像是 work 中 unit 的数量，即最外层 for 循环的数量。
// cost_per_unit: 对每个 unit 的 cpu 循环的一个估计。

// work: 一个可调用对象，work的调用应该是这样的 work(int64, int64)
```



**Shard 的实现源码：**[地址](https://github.com/tensorflow/tensorflow/blob/76b2c0630b850694d0fb3dd0b670b4d9e75d9513/tensorflow/core/util/work_sharder.cc) 地址如果失效，就去 `tensorflow/core/util/work_sharder.cc` 找

> 将work 分块执行，[0, limit) 变成 [0,block_size),  [block_size, 2*block_size) 这么一块一块。
>
> num_shards = total * cost_per_unit / 10000  为了理解 cost_per_unit 可以只关心这一部分
>
> 从这个部分可以看出，如果cost_per_unit 的运算量很大的话，tensorflow 会多分几块，那么问题来了，分成多少是合适的呢？
>
> block_size = (total + num_shards - 1) / num_shards 。num_shards 要分成几块。

```c++
#include "tensorflow/core/util/work_sharder.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

void Shard(int max_parallelism, thread::ThreadPool* workers, int64 total,
           int64 cost_per_unit, std::function<void(int64, int64)> work) {
  CHECK_GE(total, 0);
  if (total == 0) {
    return;
  }
  if (max_parallelism <= 1) {
    // Just inline the whole work since we only have 1 thread (core).
    work(0, total);
    return;
  }
  if (max_parallelism >= workers->NumThreads()) {
    workers->ParallelFor(total, cost_per_unit, work);
    return;
  }
  cost_per_unit = std::max(1LL, cost_per_unit);
  // We shard [0, total) into "num_shards" shards.
  //   1 <= num_shards <= num worker threads
  //
  // If total * cost_per_unit is small, it is not worth shard too
  // much. Let us assume each cost unit is 1ns, kMinCostPerShard=10000
  // is 10us.
  static const int64 kMinCostPerShard = 10000;
  const int num_shards =
      std::max<int>(1, std::min(static_cast<int64>(max_parallelism),
                                total * cost_per_unit / kMinCostPerShard));

  // Each shard contains up to "block_size" units. [0, total) is sharded
  // into:
  //   [0, block_size), [block_size, 2*block_size), ...
  // The 1st shard is done by the caller thread and the other shards
  // are dispatched to the worker threads. The last shard may be smaller than
  // block_size.
  const int64 block_size = (total + num_shards - 1) / num_shards;
  CHECK_GT(block_size, 0);  // total > 0 guarantees this.
  if (block_size >= total) {
    work(0, total);
    return;
  }
  const int num_shards_used = (total + block_size - 1) / block_size;
  BlockingCounter counter(num_shards_used - 1);
  for (int64 start = block_size; start < total; start += block_size) {
    auto limit = std::min(start + block_size, total);
    workers->Schedule([&work, &counter, start, limit]() {
      work(start, limit);        // Compute the shard.
      counter.DecrementCount();  // The shard is done.
    });
  }

  // Inline execute the 1st shard.
  work(0, std::min(block_size, total));
  counter.Wait();
}

}  // end namespace tensorflow
```



