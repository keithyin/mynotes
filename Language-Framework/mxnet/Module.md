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

 

## 当调用Module.bind 的时候，后端发生了什么？

**首先看创建mod的时候：**

```python
#接口
def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None)

####### 实例
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])
# 传入了 data_names 和 label_names 还有一个 state_names ，只是没传进去
# inputs_names 就是 data_names 加 label_names
# arg_names: 是 net.list_arguments() 即，计算net的值所需要的所有的 Varaible。
# _param_names 这个表示arg_names 中去掉 input_names 的 name，即不将input涉及的
# Symbol 作为模型 参数。
# self._fixed_param_names = fixed_param_names 用来指明哪些参数不需要更新
# self._aux_names = symbol.list_auxiliary_states()
# self._data_names = data_names
# self._label_names = label_names
# self._state_names = state_names
# self._output_names = symbol.list_outputs()

#############先不管这些params 是啥，继续看bind################################
# self._arg_params = None  
# self._aux_params = None
# self._params_dirty = False

##可以看出，创建 mod 对象的时候，也就是对几个对象属性赋了值。
```

**mod.bind**

```python
#接口
def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write')
###实例
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# 这个函数将 Symbol 与 executor bind起来。
# 这个函数中调用了这么对象 ，我们看一下这个对象的源码， 这个代码返回了一个 DPEG 对象
self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                     self._work_load_list, self._data_shapes,  self._label_shapes,                            self._param_names, for_training, inputs_need_grad, shared_group,                          logger=self.logger, fixed_param_names=self._fixed_param_names,
                     grad_req=grad_req,  state_names=self._state_names)

# mod 拿到了这个对象之后，做了啥呢？
if shared_module is not None:
    self.params_initialized = True
    self._arg_params = shared_module._arg_params
    self._aux_params = shared_module._aux_params
elif self.params_initialized:
    # if the parameters are already initialized, we are re-binding
    # so automatically copy the already initialized params
    self._exec_group.set_params(self._arg_params, self._aux_params)
else:
    assert self._arg_params is None and self._aux_params is None
    param_arrays = [ nd.zeros(x[0].shape, dtype=x[0].dtype)
                          for x in self._exec_group.param_arrays ]
    self._arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)}

    aux_arrays = [nd.zeros(x[0].shape, dtype=x[0].dtype)
                                for x in self._exec_group.aux_arrays]
    self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)}
    
## 看到这里，看到了 NDArray 的影子，mod 给每个 param_params 都创建了一个 NDArray，
## Symbol 变 NDArray 咯
## mod 中的 _arg_params 存放的是 name:NDArray，是 param 的 NDArray，为什么叫 _arg_params
## 而不叫_param_prarams。
# 名字是 *_params 的都是代表的 name:NDArray 键值对
# 名字是 *_arrays 存放的是 NDArray
```



**mod.init_params**

> 看完了 bind，bind 时候是将所有的 arg_params 初始化为 0 的 NDArray，而且还保存了一个 DPEG 对象，还不是道是干嘛的。

```python
def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False)
# 初始化 arg_params 和 aux_params ,调用 initializer，这里嘛，不同的参数可以调用不同的方法

for name, arr in self._arg_params.items():
    desc = InitDesc(name, attrs.get(name, None))
    _impl(desc, arr, arg_params)

for name, arr in self._aux_params.items():
    desc = InitDesc(name, attrs.get(name, None))
    _impl(desc, arr, aux_params)

self.params_initialized = True
self._params_dirty = False

# copy the initialized parameters to devices，拷贝到 GPU上去。。
self._exec_group.set_params(self._arg_params, self._aux_params)
```





**mod.forward**

> 进行前向传导

```python
def forward(self, data_batch, is_train=None):
    assert self.binded and self.params_initialized
    
    # 调用的 executor 的 forward 方法
    # 还记得在 mod.init_params 的时候，我们已经设置过 extc_group 中 模型参数的值了。
    self._exec_group.forward(data_batch, is_train)
```



**mod.update_metric**

> 更新 metric 的值， 除了更新一下metric 的值，啥也不干



**mod.backward**

> 反向传导

```python
def backward(self, out_grads=None):
    assert self.binded and self.params_initialized
    # 反向传导，用的 也是 exec，看来这玩意是核心计算单元啊
    self._exec_group.backward(out_grads=out_grads)
```

还记得在 `DataParallelExecutorGroup` 初始化函数时有一个成员属性  `self.grad_arrays` 不出所料的话，这个属性就是保存 反向传导 计算出的梯度的地方。

`DataParallelExecutorGroup` 中有一个 `_collect_arrays` 方法，这个方法是用来：

* 从 `executor` 中取 `grad_arrays` 的值，赋给 `self.grad_arrays` 属性

> 遗留问题：在 `executor` 中没有找到 给 grad_arrays 赋值的语句







**mod.update**

> 模型参数更新阶段了

```python
# 看了这个方法，我们 获得的信息是：
# self._exec_group.param_arrays 模型参数在这
# self._exec_group.grad_arrays  backward 求完的梯度在这
def update(self):
    assert self.binded and self.params_initialized and self.optimizer_initialized

    self._params_dirty = True
    if self._update_on_kvstore:
        _update_params_on_kvstore(self._exec_group.param_arrays,
                                  self._exec_group.grad_arrays,
                                  self._kvstore)
    else:
        # 看这个最简单的方法吧
        # self._updater 是通过optimizer获取的
        _update_params(self._exec_group.param_arrays,
                       self._exec_group.grad_arrays,
                       updater=self._updater,
                       num_device=len(self._context),
                       kvstore=self._kvstore)

        
def _update_params(param_arrays, grad_arrays, updater, num_device,
                   kvstore=None):
    """Perform update of param_arrays from grad_arrays not on kvstore."""
    for index, pair in enumerate(zip(param_arrays, grad_arrays)):
        # 两个都是 NDArray 列表
        arg_list, grad_list = pair
        if grad_list[0] is None:
            continue
        # 先不管 kvstore
        if kvstore:
            # push gradient, priority is negative index
            kvstore.push(index, grad_list, priority=-index)
            # pull back the sum gradients, to the same locations.
            kvstore.pull(index, grad_list, priority=-index)
        # 
        for k, p in enumerate(zip(arg_list, grad_list)):
            # w: arg, g: grad
            w, g = p
            # 这边进行更新参数咯
            updater(index*num_device+k, g, w)
```





**DataParallelExecutorGroup：**

> 看名字是 数据并行

```python
def __init__(self, symbol, contexts, workload, data_shapes, label_shapes, param_names,
                 for_training, inputs_need_grad, shared_group=None, logger=logging,
                 fixed_param_names=None, grad_req='write', state_names=None)

# self.param_names = param_names
# self.arg_names = symbol.list_arguments()
# self.aux_names = symbol.list_auxiliary_states()
# self.grad_req = {} ，用来保存模型中所有的 参数的 梯度需求情况。
# self.batch_size = None
# self.slices = None
# self.execs = []
# self._default_execs = None
# self.data_arrays = None
# self.label_arrays = None
# self.param_arrays = None
# self.state_arrays = None
# self.grad_arrays = None
# self.aux_arrays = None
# self.input_grad_arrays = None

# self.data_shapes = None
# self.label_shapes = None
# self.data_names = None
# self.label_names = None
# self.data_layouts = None
# self.label_layouts = None
# self.output_names = self.symbol.list_outputs()
# self.output_layouts = [DataDesc.get_batch_axis(self.symbol[name].attr('__layout__'))
#                        for name in self.output_names]
# self.num_outputs = len(self.symbol.list_outputs())

# self.bind_exec(data_shapes, label_shapes, shared_group)
```



## 总结

> mxnet 是把计算图部分 编译成一个函数，然后前向，后向，更新的部分，是命令式编程。

* 核心计算模块： `Executor`
  * 保存着模型参数
  * 执行前向计算，后向计算。
* 参数更新模块： `Optimizer`
  * 负责参数更新
* 助手模块： `Module`
  * 调用 `Initilizer` 初始化 模型参数，保存着初始化的 `Variable`
  * 更方便的 前向，后向，更新操作（帮助我们调用了 executor）




## 模型的保存和加载

**保存**

```python
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)

```



**加载**

```python
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
assert sym.tojson() == net.tojson()

# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)

## 或
mod = mx.mod.Module(symbol=sym)
mod.fit(train_iter,
        num_epoch=8,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)
```

## 参考资料

[http://mxnet.io/tutorials/basic/module.html](http://mxnet.io/tutorials/basic/module.html)

