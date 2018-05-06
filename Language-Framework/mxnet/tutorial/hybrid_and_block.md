# Gluon : Block 和 HybridBlock

新版 `mxnet` 中有两个类，我们可以继承这两个类来构建我们复杂神经网络，那么这两个类的关系是什么呢？



## Block

```python
class Block(object):
    
    def __call__(self, *args):
        """Calls forward. Only accepts positional arguments."""
        return self.forward(*args)
    
    def forward(self, *args):
        """Overrides to implement forward computation using `NDArray`. Only
        accepts positional arguments.

        Parameters
        ----------
        *args : list of NDArray
            Input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError
```

* 神经网络层和模型的基类，自己写的模型也应该继承这个类。
* `forward` 方法定义了前向传导过程



## HybridBlock

```python
class HybridBlock(Block):
    """`HybridBlock` supports forwarding with both Symbol and NDArray."""
    def forward(self, x, *args):
        """Defines the forward computation. Arguments can be either
        `NDArray` or `Symbol`."""
        if isinstance(x, NDArray):
            with x.context as ctx:
                if self._active:
                    return self._call_cached_op(x, *args)
                try:
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                except DeferredInitializationError:
                    self.infer_shape(x, *args)
                    for i in self.collect_params().values():
                        i._finish_deferred_init()
                    params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                return self.hybrid_forward(ndarray, x, *args, **params)

        assert isinstance(x, Symbol), \
        "HybridBlock requires the first argument to forward be either " \
        "Symbol or NDArray, but got %s"%type(x)
        params = {i: j.var() for i, j in self._reg_params.items()}
        with self.name_scope():
            return self.hybrid_forward(symbol, x, *args, **params)
  
    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.
        """
        # pylint: disable= invalid-name
        raise NotImplementedError
```

* HybridBlock 可以用来 推断 两个类型的数据 `Symbol` 和 `NDArray`
* Block 只能用来推断 `NDArray`
* HybriBlock 是用来与 符号编程 转换的
* Block 设计目的就是 命令式。


**看hybrid_forward的参数**

```python
def hybrid_forward(self, F, x, *args, **kwargs)
```

* `F` 用来表示 是 命令式运算符 还是 符号运算符
  * `F=mx.nd` 或者 `F=mx.sym`
* ​




## Block 与 HybridBlock

在写我们模型的时候，这两个都可以继承：

* Block : 主要重写 `__init__` 和 `forward` 方法
* HybridBlock ： 主要重写 `__init__` 和 `hybrid_forward` 方法




## Block 细节

**name_scope是干嘛的？**

* 给 **Block** 和 **Parameter** 的名字前面**加前缀**

[http://zh.gluon.ai/chapter_gluon-basics/block.html#%E4%BD%BF%E7%94%A8-nn.Block-%E6%9D%A5%E5%AE%9A%E4%B9%89](http://zh.gluon.ai/chapter_gluon-basics/block.html#%E4%BD%BF%E7%94%A8-nn.Block-%E6%9D%A5%E5%AE%9A%E4%B9%89)

看一下 `nn.Block` 源码

```python
class Block(object):
    def __init__(self, prefix=None, params=None):
        self._empty_prefix = prefix == ''
        # _prefix 是个 string， _params 是个 ParameterDict
        self._prefix, self._params = _BlockScope.create(prefix, params, self._alias())
        
        # 设置 Block 的那么
        self._name = self._prefix[:-1] if self._prefix.endswith('_') else self._prefix
        # 创建一个 BlockScope 对象
        self._scope = _BlockScope(self)
        
        # 用于存放 子 !!Block!! 的
        self._children = []
    
    def name_scope(self):
        return self._scope
    
```

可见， `with self.name_scope()` 仅仅是 将 `Block` 初始化时候 创建的 `_BlockScope` 拿出来用而已。 对于如何使用 `name_scope` ，在 创建模型块之前 `with` 一下即可。



## `_BlockScope`

`_BlockScope` 相当于 tf 的 `VariableScope`， `NameManager` 等价于 tf 的 `NameScope`。

上下文管理器。

```python
class _BlockScope(object):
    """Scope for collecting child `Block` s."""
    # 用来保存当前 的 Scope
    _current = None

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """Creates prefix and params for new `Block`."""
        current = _BlockScope._current
        if current is None:
            if prefix is None:
                prefix = _name.NameManager.current.get(None, hint) + '_'
            if params is None:
                # 这里可以看出，参数是在 BlockScope 下创建的。
                params = ParameterDict(prefix)
            else:
                params = ParameterDict(params.prefix, params)
            return prefix, params

        if prefix is None:
            count = current._counter.get(hint, 0)
            prefix = '%s%d_'%(hint, count)
            current._counter[hint] = count + 1
        if params is None:
            parent = current._block.params
            params = ParameterDict(parent.prefix+prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        return current._block.prefix+prefix, params
    # 执行 enter 的时候，_current 替换
    def __enter__(self):
        if self._block._empty_prefix:
            return
        self._old_scope = _BlockScope._current
        _BlockScope._current = self
        self._name_scope = _name.Prefix(self._block.prefix)
        # name scope 手动进入。
        self._name_scope.__enter__()
        return self
	# 执行 exit 的时候，_current 换回
    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        # 退出 name_scope
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
        _BlockScope._current = self._old_scope
```



## ParameterDict

用来存放 参数的 类。用 `OrderedDict()` 保存着 `Block` 下的 `Parameter`。



```python
class ParameterDict(object):
    """A dictionary managing a set of parameters.

    Parameters
    ----------
    prefix : str, default ``''``
        The prefix to be prepended to all Parameters' names created by this dict.
    shared : ParameterDict or None
        If not ``None``, when this dict's :py:meth:`get` method creates a new parameter,
        will first try to retrieve it from "shared" dict. Usually used for sharing
        parameters with another Block.
        用来共享 Parameter 的。
    """
    def __init__(self, prefix='', shared=None):
        self._prefix = prefix
        self._params = OrderedDict()
        self._shared = shared
    def get(self, name, **kwargs):
    """
    先找 当前 ParameterDict 有没有，如果有，返回，没有，去shared 中找，还没有，创建一个。
    Parameters
    ----------
    name : str
        Name of the desired Parameter. It will be prepended with this dictionary's
        prefix.
    **kwargs : dict
        The rest of key-word arguments for the created :py:class:`Parameter`.

    Returns
    -------
    Parameter
        The created or retrieved :py:class:`Parameter`.
    """
      name = self.prefix + name
      param = self._get_impl(name)
      if param is None:
          param = Parameter(name, **kwargs)
          self._params[name] = param
      else:
          for k, v in kwargs.items():
              if hasattr(param, k) and getattr(param, k) is not None:
                  assert v is None or v == getattr(param, k), \
                      "Cannot retrieve Parameter %s because desired attribute " \
                      "does not match with stored for attribute %s: " \
                      "desired %s vs stored %s." % (
                          name, k, str(v), str(getattr(param, k)))
              else:
                  setattr(param, k, v)
      return param
```



## 总结

**Block**

*  都有自己独立的 `ParameterDict` 对象。
* 子 `Block` 保存在 `self._children` 中。 
* 参数保存 `ParameterDict` 中。



**参数加载**

* Block 中有个 参数加载的入口，但是实际上参数加载是由 `ParameterDict` 负责的。







