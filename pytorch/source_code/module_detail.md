# pytorch: Module

**Module** 是 `pytorch` 提供的一个基类，每次我们要 搭建 自己的神经网络的时候都要继承这个类，继承这个类会使得我们 搭建网络的过程变得异常简单。

本文主要关注 `Module` 类的内部是怎么样的。



## 初始化方法中做了什么

```python
def __init__(self):
    self._backend = thnn_backend
    self._parameters = OrderedDict()
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self._forward_pre_hooks = OrderedDict()
    self._modules = OrderedDict()
    self.training = True
```

这是 `Module` 的初始化方法：

* `self._parameters` 用来存放注册的 `Parameter` 对象
* `self._buffers` 用来存放注册的 `Buffer` 对象。（不存在Buffer类）
* `self._modules` 用来保存注册的 `Module` 对象。
* `self.training` 标志位，用来表示是不是在 `training` 状态下
* `...hooks` 用来保存 注册的 `hook`



## `__setattr__` 与 `__getattr__`

> `__setattr__` 每次给属性赋值的时候，都会调用这个方法。



`__setattr__` 的代码比较多，我们一点一点看。



* `remove_from` ：工具函数， 用来从 `self.__dict__, self._buffers, self._modules` 中删除对象。



第一种情况： `value` 的类型是 `Paramter`

* 从 三大 字典中将 同名的 对象删掉
* 然后，注册 `paramter`

第二种情况：  `value`不是 `Parameter`对象，  `name `在 `self._parameter` 中

* `self._parameters[name] = None`



**已经考虑了 `value` 是 `Parameter`对象，剩下的就是考虑 `value` 为 `buffer`或 `Module` 了**



第三种情况：`value`不是 `Parameter`对象， `value` 为 `Module` 对象

* 从三大字典里面移除同名 对象
* 然后直接向 `self._modules` 字典里添加 `value`



第四种情况：`value`不是`Parameter`对象， `value`不为 `Module`对象， 但是 `name` 在 `self._modules` 里

* `self._modules[name]=None`



第五种情况：`value`不是`Parameter`对象， `value`不为 `Module`对象, `name` 存在 `self._buffers` 里

*  `self._buffers[name]=None`



最后一种情况： 就是 普通的属性了。

```python
def __setattr__(self, name, value):
    def remove_from(*dicts):
        for d in dicts:
            if name in d:
                del d[name]

    params = self.__dict__.get('_parameters')
    
    if isinstance(value, Parameter):
        if params is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call")
        remove_from(self.__dict__, self._buffers, self._modules)
        self.register_parameter(name, value)
    elif params is not None and name in params:
        if value is not None:
            raise TypeError("cannot assign '{}' as parameter '{}' "
                            "(torch.nn.Parameter or None expected)"
                            .format(torch.typename(value), name))
        self.register_parameter(name, value)
    else:
        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(torch.nn.Module or None expected)"
                                .format(torch.typename(value), name))
            modules[name] = value
        else:
            buffers = self.__dict__.get('_buffers')
            if buffers is not None and name in buffers:
                if value is not None and not torch.is_tensor(value):
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(torch.Tensor or None expected)"
                                    .format(torch.typename(value), name))
                buffers[name] = value
            else:
                object.__setattr__(self, name, value)
```



> `__getattr__` :  当获取 `self.__dict__` 中没有的键所对应的值的时候，就会调用这个方法
>
> 因为 `parameter, module, buffer` 的键值对存在与 `self._parameters, self._modules, self.buffer` 中，所以，当想获取这些 值时， 就会调用这个方法。

```python
def __getattr__(self, name):
    if '_parameters' in self.__dict__:
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            return _parameters[name]
    if '_buffers' in self.__dict__:
        _buffers = self.__dict__['_buffers']
        if name in _buffers:
            return _buffers[name]
    if '_modules' in self.__dict__:
        modules = self.__dict__['_modules']
        if name in modules:
            return modules[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(
        type(self).__name__, name))
```





## register_parameter

**向模型中注册 Parameter**

```python
def register_parameter(self, name, param):
    """Adds a parameter to the module.

    The parameter can be accessed as an attribute using given name.
    """
    if '_parameters' not in self.__dict__:
        raise AttributeError(
            "cannot assign parameter before Module.__init__() call")
    if param is None:
        self._parameters[name] = None
    elif not isinstance(param, Parameter):
        raise TypeError("cannot assign '{}' object to parameter '{}' "
                        "(torch.nn.Parameter or None required)"
                        .format(torch.typename(param), name))
    elif param.grad_fn:
        raise ValueError(
            "Cannot assign non-leaf Variable to parameter '{0}'. Model "
            "parameters must be created explicitly. To express '{0}' "
            "as a function of another variable, compute the value in "
            "the forward() method.".format(name))
    else:
        self._parameters[name] = param
```





## Module.training 标志 如何影响 前向过程

从`nn.Dropout` 来看 `Module.training`

```python
class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)
```

可以看出，在`forward` 过程中，直接获取，父类的`training`的值。



我们 通常通过 `module.train()` 和 `module.eval()` 来切换模型的 训练测试阶段。

```python
def train(self, mode=True):
    """Sets the module in training mode.
    This has any effect only on modules such as Dropout or BatchNorm.
    """
    self.training = mode
    
    for module in self.children():
        # 递归调用子模块 train 函数， 来设定所有 module 的 training 值。
        module.train(mode)
        return self
```





## 参考资料

[https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py) 

