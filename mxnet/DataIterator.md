# mxnet 学习笔记（四）：DataIterator

> mxnet 中的 Data Iterator 跟python中 可迭代对象差不多



`MXNet` 中，每次调用 `next()`，`data iterators` 就会返回一个 `mini-batch` 数据。一个 `mini-batch` 通常包含 `n`（`batch-size`） 个训练样本和其对应的标签。 当没有数据可以读取的时候，`iterator` 会抛出一个 `StopIteration`异常。



**DataDesc**

----



## 自定义 Iterator

