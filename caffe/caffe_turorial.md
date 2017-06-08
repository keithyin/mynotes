# caffe tutorial

## 核心概念

* Blob：caffe中传递的数据的wrapper，数值上像是`ndarray`
* Layer: 代表神经网络中的一层（卷积层，池化层，全连接层，etc）
* Solver：用于优化网络的（执行forward，backward，梯度更新任务）



## forward

* 定义caffe模型的话，只需在`.prototxt`文件中定义好就行了
* 学习到的模型 被保存到 `.caffemodel`文件中



**一个.prototxt例子**

```protobuf
name: "LogReg"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "input_leveldb"
    batch_size: 64
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
}
```



**.prototxt文件中只需定义前向过程就可以了，反传的过程不需要care**



* 如何定义`input`和`loss`
  * 输入层没有 `bottom`
  * `loss` 由层的 `loss_weight`参数指定
    * 普通层的 `loss_weight` 参数为`0`
    * 后缀有`Loss`的层，`loss_weight`默认值为1



**.prototxt文件中的layer定义关键字解释**

* `bottom`: 层的输入
* `top`: 层的输出
* `name`: 层的名字（可以用来干啥？）
* `type`: 是什么层（卷积，全连接，etc）
* ***_param
  * 用于设置层的参数




## caffe中几个关键文件

* `model_name.prototxt`
  * 用于保存定义的网络网络结构
* `.caffemodel`
  * 保存的是模型的参数
* ​



