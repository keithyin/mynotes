# tensorflow flold : dynamic batching

> 专门用来搞定 变长数据的



## Blocks

`tensorflow-fold` 的一个基本元素就是 `blocks`：

* `block` 是个**函数**， 用来将 python 对象转成 `tensorflow`的 `tensor` 对象



```python
import tensorflow_fold as td
scalar_block = td.Scalar()
vector3_block = td.Vector(3) # 3元素的向量

# 使用 td.Record() 将简单的 Block 组合起来
record_block = td.Record({'foo': scalar_block, 'bar': vector3_block})
```





## 参考资料

[https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/quick.ipynb](https://github.com/tensorflow/fold/blob/master/tensorflow_fold/g3doc/quick.ipynb)

