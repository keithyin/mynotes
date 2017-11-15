# pytorch : DataLoader

`pytorch`  数据加载部分的 接口可以说是现存 深度学习框架中设计的最好的， 给了我们足够的灵活性。本博文就对 pytorch 的多线程加载 模块（`DataLoader`） 进行源码上的注释。

## 输入流水线

 `pytorch` 的输入流水线的操作顺序是这样的：

* 创建一个 Dataset 对象
* 创建一个 DataLoader 对象
* 不停的 循环 这个 DataLoader 对象 

```python
dataset = MyDataset()
dataloader = DataLoader(dataset)
num_epoches = 100
for epoch in range(num_epoches):
    for data in dataloader:
        ....
```

在之前文章也提到过，如果现有的 `Dataset` 不能够满足需求，我们也可以自定义 `Dataset`，通过继承 `torch.utils.data.Dataset`。在继承的时候，需要 `override ` 三个方法。

* `__init__`： 用来初始化数据集
* `__getitem__`
* `__len__`

从本文中，您可以看到 `__getitem__` 和 `__len__` 在 `DataLoader` 中是如何被使用的。

## DataLoader

从`DataLoader` 看起，下面是源码。为了方便起见，采用在源码中添加注释的形式进行解读。

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
                    # dataset.__len__() 在 Sampler 中被使用。
                    # 目的是生成一个 长度为 len(dataset) 的 序列索引（随机的）。
                    sampler = RandomSampler(dataset)
                else:
                    # dataset.__len__() 在 Sampler 中被使用。
                    # 目的是生成一个 长度为 len(dataset) 的 序列索引（顺序的）。
                    sampler = SequentialSampler(dataset)
            # Sampler 是个迭代器，一次之只返回一个 索引
            # BatchSampler 也是个迭代器，但是一次返回 batch_size 个 索引
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
```



```python
# 以下两个代码是等价的
for data in dataloader:
    ...
# 等价与
iterr = iter(dataloader)
while True:
    try:
        next(iterr)
    except:
        break
```

在 `DataLoader` 中，`iter(dataloader)` 返回的是一个 `DataLoaderIter` 对象， 这个才是我们一直 `next`的 对象。

下面会先介绍一下 几个  `Sampler`， 然后介绍 核心部分 `DataLoaderIter`。



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
        # 这里的 sampler 是 RandomSampler 或者 SequentialSampler
        # 他们每一次吐出一个 idx
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



1. `self.index_queue` 中存放是 `(batch_idx, sample_indices)` ，其中 `batch_idx` 是个 `int` 值， `sample_indices` 是个 `list` ， 存放了 组成 `batch` 的 `sample indices`。
2. `self.data_queue` 中存放的是 `(batch_idx, samples)`, 其中 `samples` 是 一个 `mini-batch` 的样本
3. `self.send_idx` 表示：这次 放到 `self.index_queue` 中的 `batch_id`
4. `self.rcvd_idx` 表示：这次要取的 `batch_id`
5. `self.batches_outstanding` 表示：




```python
class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        # loader 是 DataLoader 对象
        self.dataset = loader.dataset
        # 这个留在最后一个部分介绍
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        # 表示 开 几个进程。
        self.num_workers = loader.num_workers
        # 是否使用 pin_memory
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()
		
        # 这样就可以用 next 操作 batch_sampler 了
        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            # 用来放置 batch_idx 的队列，其中元素的是 一个 list，其中放了一个 batch 内样本的索引
            self.index_queue = multiprocessing.SimpleQueue()
            # 用来放置 batch_data 的队列，里面的 元素的 一个 batch的 数据
            self.data_queue = multiprocessing.SimpleQueue()
            
            # 当前已经准备好的 batch 的数量（可能有些正在 准备中）
            # 当为 0 时， 说明， dataset 中已经没有剩余数据了。
            # 初始值为 0, 在 self._put_indices() 中 +1,在 self.__next__ 中减一
            self.batches_outstanding = 0 
            self.shutdown = False
            # 用来记录 这次要放到 index_queue 中 batch 的 idx
            self.send_idx = 0
            # 用来记录 这次要从的 data_queue 中取出 的 batch 的 idx
            self.rcvd_idx = 0
            # 因为多线程，可能会导致 data_queue 中的 batch 乱序
            # 用这个来保证 batch 的返回 是 idx 升序出去的。
            self.reorder_dict = {}
            # 这个地方就开始 开多进程了，一共开了 num_workers 个进程
            # 执行 _worker_loop ， 下面将介绍 _worker_loop
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            # 初始化的时候，就将 2*num_workers 个 (batch_idx, sampler_indices) 放到 index_queue 中。
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            # 说明没有 剩余 可操作数据了， 可以停止 worker 了
            self._shutdown_workers()
            raise StopIteration

        while True:
            # 这里的操作就是 给 乱序的 data_queue 排一排 序
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            # 一个 batch 被 返回，batches_outstanding -1
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            # 返回的时候，再向 indice_queue 中 放下一个 (batch_idx, sample_indices)
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        # 放下一个 (batch_idx, sample_indices)
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                # shutdown 的时候， 会将一个 None 放到 index_queue 中
                # 如果 _worker_loop 获得了这个 None， _worker_loop 将会跳出无限循环，将会结束运行
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()
```



## `__worker_loop`

这部分是 多进程 执行的代码：他从`index_queue` 中 取索引，然后处理数据，然后再将 处理好的 batch 数据放到 `data_queue` 中。

```python
def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True
    
    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            # 想 data_queue 中放 None
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            # 这里就可以看到 dataset.__getiterm__ 的作用了。
            # 传到 collate_fn 的数据是 list of ...
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
```



## collate_fn

* 我们 `__getiterm__` 经常返回的是 （img_tensor, label）, 


* 所以 放入 `collate_fn` 的 参数就是 `[(img_tensor, label), ....]` . 
* `batch[0]` 就是 `(img_tensor, label）` ， 也就是  `collections.Sequence` 类型。



```python
def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            # 计算 batch 中所有 元素的个数 
            numel = sum([x.numel() for x in batch])
            # 没有找到对应的 api 。。。。。。
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))
```







## 总结

* `data_queue` 中最多有 `2*num_worker` 个 `batch`
* ​











## Queue的特点

* 当里面没有数据时： `queue.get()` 会阻塞
* 当数据满了: `queue.put()` 会阻塞

