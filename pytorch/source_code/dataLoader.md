# pytorch : DataLoader



## DataLoader

```python
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, 
                 batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, 
                 drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
```



## RandomSampler, SequentialSampler, BatchSampler

首先，是 `RandomSampler`， `iter(randomSampler)` 会返回一个可迭代对象，这个可迭代对象 每次 `next` 都会输出当前要采样的 `index`，`SequentialSampler`也是一样，只不过她产生的 `index` 是**顺序**的

```python
class RandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return len(self.data_source)
```



`BatchSampler` 是一个普通 `Sampler` 的  `wrapper`， 普通`Sampler` 一次仅产生一个 `index`， 而 `BatchSampler` 一次产生一个 `batch` 的 `indices`。

```python
class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```





## DataLoaderIter

这里是对下面代码的解读：

1. `self.index_queue` 中存放是 `(batch_id, sample_indices)` ，其中 `batch_id` 是个 `int` 值， `sample_indices` 是个 `list` ， 存放了 组成 `batch` 的 `sample indices`。
2. `self.data_queue` 中存放的是 `(batch_id, samples)`, 其中 `samples` 是 一个 `mini-batch` 的样本
3. `self.send_idx` 表示：这次 放到 `self.index_queue` 中的 `batch_id`
4. `self.rcvd_idx` 表示：这次要取的 `batch_id`
5. `self.batches_outstanding` 表示：



**DataLoaderIter** 基本执行流程是：

* 初始化操作： 先往 `self.index_queue` 中放 `2×num_worker` 个 `(batch_id, sample_indices)` 

```python
for _ in range(2 * self.num_workers):
    self._put_indices()
```



* （多进程一起操作）然后，`_worker_loop` 就会从 `index_queue` 取出放入 的 `(batch_id, sample_indeices)` ，用这个搞出来真正的数据，放入到 `self.data_queue` 中。

```python
def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
```



*  然后 `__next__` 的时候干了两件事：
  * 将当前要取 的 `batch_id` 从 `data_queue` 中取出来
  * 执行一次  `put_indices` 操作。

```python
if self.rcvd_idx in self.reorder_dict:
    batch = self.reorder_dict.pop(self.rcvd_idx)
    return self._process_next_batch(batch)

if self.batches_outstanding == 0:
    # 表示所有的 batch 均以 拿出，已经没有剩余数据了，一次 epoch 结束
    self._shutdown_workers()
    raise StopIteration

# 这部分代码的解释是：
# 因为 data_queue 中的数据是 多进程往里 放的， 所以里面存储的 batch_id 是无序的。
# 所以 外面需要一个 reorder_dict 来重新处理一下 data_queue 中无序的 batch_id
while True:
    assert (not self.shutdown and self.batches_outstanding > 0)
    idx, batch = self.data_queue.get()
    self.batches_outstanding -= 1
    if idx != self.rcvd_idx:
        # store out-of-order samples
        self.reorder_dict[idx] = batch
        continue
    return self._process_next_batch(batch)
```





## collate_fn







## 总结

* 内存中最多有 `2*num_worker` 个 `batch`
* ​



 







## Queue的特点

* 当里面没有数据时： `queue.get()` 会阻塞
* 当数据满了: `queue.put()` 会阻塞

