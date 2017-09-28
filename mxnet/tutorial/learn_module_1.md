# mxnet : Module 类（一）

**Module** 是 mxnet 提供给用户的一个高级封装的类。有了它，我们可以很容易的来训练模型。



**Module 包含以下单元的一个 wraper**

* `symbol` : 用来表示网络前向过程的 `symbol`。
* `optimizer`: 优化器，用来更新网络。
* `exec_group`: 用来执行 前向和反向计算。
* ...



**所以 Module 可以帮助我们做**

* 前向计算，（由 `exec_group` 提供支持）
* 反向计算，（由 `exec_group` 提供支持）
* 更新网络，（由 `optimizer` 提供支持）
* ...



## 一个 Demo

下面来看 [MXNET 官网上提供的一个 Module 案例](https://mxnet.incubator.apache.org/tutorials/basic/module.html)

第一部分：准备数据

```python
import logging
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
import numpy as np

fname = mx.test_utils.download('http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')
data = np.genfromtxt(fname, delimiter=',')[:,1:]
label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

batch_size = 32
ntrain = int(data.shape[0]*0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

```

第二部分：构建网络

```python
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')
mx.viz.plot_network(net)

```



第三部分：创建Module

```python
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# 通过data_shapes 和 label_shapes 推断其余参数的 shape，然后给它们分配空间
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# 初始化模型的参数
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# 初始化优化器，优化器用来更新模型
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
metric = mx.metric.create('acc')
# train 5 epochs, i.e. going over the data iter one pass
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # 前向计算
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # 反向传导
        mod.update()                            # 更新参数
    print('Epoch %d, Training %s' % (epoch, metric.get()))
```

关于 `bind` 的参数：

* `data_shapes` : list of (str, tuple), **str 是 数据 Symbol 的名字**，tuple是 mini-batch 的形状，所以一般参数是`[('data', (64, 3, 224, 224))]`
* `label_shapes`: list of (str, tuple)，**str 是 标签 Symbol 的名字**，tuple是 mini-batch 标签的形状，一般 分类任务的 参数为 `[('softmax_label'),(64,)]`
* 为什么上面两个参数都是 `list` 呢？ 因为可能某些网络架构，不止一个 数据，不止一种 标签。



关于 `forward`的参数

* `data_batch` : 一个 `mx.io.DataBatch`-like 对象。只要一个对象，可以 `.data`返回 mini-batch 训练数据， `.label` 返回相应的标签，就可以作为 `data_batch` 的实参 。
* 关于 `DataBatch`对象：`.data` 返回的是 list of NDArray（网络可能有多个输入数据），`.label` 也一样。


