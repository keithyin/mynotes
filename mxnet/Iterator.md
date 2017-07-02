# mxnet Iterator

> 数据读取

An iterator in *MXNet* should

1. 实现 `__next()__` 方法，返回一个 `DataBatch`，如果数据全部读取玩，抛出个 `StopIteration` 异常。 
2. 实现 `reset()` 方法，可以使数据从头读。
3. 写一个 `provide_data` 属性， consisting of a list of `DataDesc` objects that store the name, shape, type and layout information of the data (more info [here](http://mxnet.io/api/python/io.html#mxnet.io.DataBatch)).
4. 写一个 `provide_label` 属性， consisting of a list of `DataDesc` objects that store the name, shape, type and layout information of the label.

```python

class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration
```

