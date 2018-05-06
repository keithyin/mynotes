
# mxnet Gluon 无痛入门
沐神已经提供了一份官方的文档，为什么要写这么一篇博客： 沐神提供的中文文档质量是非常高的，[地址](https://zh.gluon.ai/install.html),但是感觉需要看一段时间才能上手 Gluon， 本博客结构模仿 pytorch 的60分钟入门博客编写，旨在用最快的时间可以直接上手Gluon。同时也可以对Gluon的几个重要模块进行一下总结，以后查询方便。 （博主水平有限，如有错误，请不吝指出。）

下面进入正题：


mxnet 新提出的 Gluon 框架是一个 动态图框架， 如果之前有接触过 pytorch 的话，那么应该对动态图框架应该非常熟悉：
* 动态图： define by run
* 静态图： define before run

静态图的代表就是 Tensorflow，我们首先定义好计算图，然后 feed 数据进行训练网络。
动态图的代表就是 chainer， pytorch 和 Gluon 了，在运行的时候定义图。在每个 mini-batch 进行训练的时候都会重新定义一次计算图。

## 安装

```shell
pip uninstall mxnet
pip install --pre mxnet-cu75 # CUDA 7.5
pip install --pre mxnet-cu80 # CUDA 8.0
```

## 数据放在哪
在神经网络中，我们有三种类别的数据：
* 样本数据（输入 和 label）
* 网络模型参数
* 网络中每层的输入 数据

在 Gluon 中，这三种类别的数据都是由 `mx.nd.NDArray` 来存储的。

但是需要注意的是：
* 模型参数记得 NDArray.attach_grad(), 因为模型参数更新的时候需要用到 梯度，attach_grad() 就是为参数梯度的存放开辟了空间，以助于参数更新的时候进行访问
* 不需要显式访问梯度的 NDArray 是不需要 attach_grad() 的



```python
from mxnet import nd
val = nd.normal(shape=(2,3)) # 在使用 ide 时，没有代码提示不代表没有，常去官网查查 API
val
```




    
    [[ 1.18392551  0.15302546  1.89171135]
     [-1.16881478 -1.23474145  1.55807114]]
    <NDArray 2x3 @cpu(0)>




```python
val_shape = val.shape # 获取 NDArray 的shape， 这操作 很 numpy
val_shape
```




    (2, 3)



mxnet.nd 中也提供了很多对 NDArray 的操作 [https://mxnet.incubator.apache.org/api/python/ndarray.html#array-creation-routines](https://mxnet.incubator.apache.org/api/python/ndarray.html#array-creation-routines)


## 自动求导
在 0.11 之前的版本中， mxnet 的 NDArray 是不支持自动求导，自动求导的支持仅存在与 mxnet 的符号编程中，但是为 Gluon（基于mxnet 的动态图框架）， mxnet 对于 NDArray 也提供了自动求导机制， 通过 **mxnet.autograd** 来支持


```python
from mxnet import nd
from mxnet import autograd

# nd.NDArray
val = nd.ones(shape=(3, 5))

w1 = nd.ones(shape=(5, 1))
b1 = nd.ones(shape=(1,))

w1.attach_grad() # 对模型参数进行 attach_grad()
b1.attach_grad() # 同上

with autograd.record(): # 使用 autograd 来记录计算图
    res = nd.dot(val, w1)
    res2 = res + b1
res2.backward() # 这里需要注意的是，如果 res2 不是标量的话，默认的操作是会对 res2 做一个 sum，然后在 backward

print(w1.grad)
print(b1.grad)
```

    
    [[ 3.]
     [ 3.]
     [ 3.]
     [ 3.]
     [ 3.]]
    <NDArray 5x1 @cpu(0)>
    
    [ 3.]
    <NDArray 1 @cpu(0)>


## 神经网络
到这里，终于可以看到 Gluon 的身影了，Gluon给我们提供了一些简洁的 高级 API，我们可以使用这个 API 快速的搭建想要的神经网络结构。

祭出神器 `mxnet.gluon`（版本 0.11 及以上 才有这个工具包）

深度学习的流水线大概有以下几个步骤：
* 搭建网络结构
* 初始化模型参数
* 训练模型参数
    * mini-batch 数据输入到网络中
    * 计算 loss
    * 反向传导得到 模型参数的梯度信息
    * 更新参数



```python
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import loss
from mxnet import gluon

class Net(nn.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)

        self.dense0 = nn.Dense(256) # 我们只需要对 层的输出维度 作说明，不需要考虑输入的维度
        self.dense1 = nn.Dense(1)   # Gluon 会帮助我们 推断出 输入的 维度

    def forward(self, x):
        return self.dense1(nd.relu(self.dense0(x)))

net = Net()

net.initialize() # 要先 initialize， 再创建 trainer
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.001}) # 优化方法
val = nd.ones(shape=(10, 100))
label = nd.ones(shape=(10,))

criterion = loss.L2Loss() # 创建一个 loss 标准

with autograd.record(): # 需要反向传导的地方记得 record 一下
    res = net(val)
    print(res)
    loss = criterion(res, label)

loss.backward() # 计算梯度

trainer.step(batch_size=10)  # 更新模型参数

print(net(val))
print(loss)

```

    
    [[-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]
     [-0.18936485]]
    <NDArray 10x1 @cpu(0)>
    
    [[-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]
     [-0.14251584]]
    <NDArray 10x1 @cpu(0)>
    
    [ 0.70729446  0.70729446  0.70729446  0.70729446  0.70729446  0.70729446
      0.70729446  0.70729446  0.70729446  0.70729446]
    <NDArray 10 @cpu(0)>



上面的事例已经将关键的 Gluon 组件一一展现了出来，包括：
* gluon.nn.Block 容器一样的概念，用来构建神经网络
* gluon.loss 各种 loss 的聚集地
* gluon.nn 有很多 层的 实现
* gluon.rnn 里面有循环神经网络的一些 Cell
* gluon.Trainer 用来辅助更新模型参数的一个辅助类
* mxnet.optimizer 里面有很多优化器

## 如何使用 GPU

当进行运算的值都处于 GPU 上时，则运算发生在 GPU 上。

**使用 ctx 来为创建的 NDArray 指定设备**


```python
import mxnet as mx
val = nd.zeros(shape=(3,),ctx=mx.gpu())
print(val)
```

    
    [ 0.  0.  0.]
    <NDArray 3 @gpu(0)>


**如何将 定义的网络的参数放到 GPU 上**


```python
net.initialize() # 利用这个函数， 里面有个 ctx 参数
```

## NDArray 与 numpy.ndarray 互相转换


```python
import numpy as np
from mxnet import nd

# numpy.ndarray --> mx.NDArray
val = np.array([1, 2, 3])
nd_val = nd.array(val) # 深复制

# NDArray --> numpy.ndarray
val_ = nd_val.asnumpy()
```

##  参考资料
[https://github.com/zackchase/mxnet-the-straight-dope/blob/7a00d1ec253129d844055870d59266a8a502f5c4/chapter06_optimization/gd-sgd.ipynb](https://github.com/zackchase/mxnet-the-straight-dope/blob/7a00d1ec253129d844055870d59266a8a502f5c4/chapter06_optimization/gd-sgd.ipynb)
[https://zh.gluon.ai/index.html](https://zh.gluon.ai/index.html)
