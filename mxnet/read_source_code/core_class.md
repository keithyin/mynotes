# 核心类型总结



## Data

* [`TBlob`](https://github.com/apache/incubator-mxnet/blob/master/include/mxnet/tensor_blob.h#L58)

> 对于 Tensor 来说，它的 shape 是不能改变的。但是深度学习 中的图模型中的 数据是需要改变 形状的，所以 TBlob 横空出世，通过 TBlob 的成员函数可以获取 其存储的数据，并转成相应的 Tensor。

* [`Tensor`](..)

> 计算的基本数据类型

* ​





## Node

`print(d.tojson())` 打印出来的东西中有个 `nodes` 列表，这个列表是模型体系结构的完整定义。

**Node包含以下几个单元**

* `op`
* `name`
* `attr`
* `inputs`



**Node中几个类**

[git地址https://github.com/dmlc/nnvm/blob/master/include/nnvm/node.h](https://github.com/dmlc/nnvm/blob/master/include/nnvm/node.h)

* `Node` : 节点类，包含 `NodeAttrs`
* `NodeEntry`： 表示 节点的 输出
* `NodeAttrs`：节点属性类，包含 `op`



## OP

[https://github.com/dmlc/nnvm/blob/master/include/nnvm/op.h](https://github.com/dmlc/nnvm/blob/master/include/nnvm/op.h)

**几个类**

* `class Op` : `Operator` 的结构，`NNVM_REGISTER_OP` 用的就是这个玩意
* `class OpMap`： 一个`Map`（键值对），`Op*` 为键， `ValueType` 为值。



## 注册Op

[https://github.com/dmlc/tvm/blob/master/include/tvm/runtime/registry.h](https://github.com/dmlc/tvm/blob/master/include/tvm/runtime/registry.h)

