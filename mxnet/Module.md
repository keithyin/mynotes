# mxnet 学习笔记（三）：Module



## 创建一个Module

常用的 module类 是 `Module`，`Module`类的构造函数中包含一下几个参数：

- `symbol`: 计算图的定义（网络的定义），即网络输出 Symbol
- `context`: 用于执行的设备，一个或多个（列表）
- `data_names` : 输入变量的变量名（即：占坑Symbol的名字）`list`。
- `label_names` :  输入标签的 变量名，`list`

```python
# 计算图声明
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')

mx.viz.plot_network(net)
##################################################

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
# softmax_label 这个Symbol 已经被mxnet自动创建，好贴心，sweet。
```



## Module 的中层 API 接口

为了训练一个 **module** 需要以下步骤：

- `bind` : 为计算分配内存。
- `init_params` : 初始化 `parameters`。Assigns and initializes parameters.
- `init_optimizer` : 初始化 优化器，默认是 `sgd`。
- `metric.create` : 通过输入标准名来创建 评价标准。
- `forward` : 前向计算
- `update_metric` : 累积计算标准。
- `backward` : 反向计算。
- `update` : 根据反向过程中对梯度的计算和安装好的 优化器 来更新参数。

```python
# allocate memory given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
metric = mx.metric.create('acc')
# train 5 epochs, i.e. going over the data iter one pass
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))
```

* 我想知道 loss 怎么定义的。。。。。。。。????



## Module 的高级接口

```python
# reset train_iter to the beginning
train_iter.reset()

# create a module
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# fit the module
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)
```



## 预测和评估

* mode.predict(val_iter) : 会返回所有的 预测结果

  ```python
  y = mod.predict(val_iter)
  assert y.shape == (4000, 26)
  ```

* mod.score(val_iter, ['mse', 'acc']): 会返回想要的基准结果

  ```python
  score = mod.score(val_iter, ['mse', 'acc'])
  print("Accuracy score is %f" % (score))
  ```



## 保存和加载

可以通过 `checkpoint` 回调来保存 `module`的参数。

```python
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)# 每个epoch保存一次
# 它怎么知道哪些是模型参数，那些不是的？？？？？


#######加载模型参数##########################################
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
assert sym.tojson() == net.tojson()

# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)


###########从间断点 继续运行程序##############################
#不用使用set_params
mod = mx.mod.Module(symbol=sym)
mod.fit(train_iter,
        num_epoch=8,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)
```

 





## 参考资料

[http://mxnet.io/tutorials/basic/module.html](http://mxnet.io/tutorials/basic/module.html)

