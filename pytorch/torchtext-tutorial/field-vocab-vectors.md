# Field, Vocab, Vectors

从上一篇博客中可以看出，`Filed` 定义了 数据样本应该怎么处理，`Field` 进行数据处理的时候会用到 `Vocab`， `Vocab` 中也包含 `Vectors`。 这篇文章文章是缕缕这三者的关系。



## 从 `Field.build_vocab` 谈起



```python
TEXT.build_vocab(train, vectors="glove.6B.100d")
```

其中

* `train` 是 `Dataset`
* `vectors` ： 是预训练好的 





## Vocab



## Vectors

里面保存着预训练的 `word vector`， 