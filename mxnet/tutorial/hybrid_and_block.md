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



## Block 与 HybridBlock

在写我们模型的时候，这两个都可以继承：

* Block : 主要重写 `__init__` 和 `forward` 方法
* HybridBlock ： 主要重写 `__init__` 和 `hybrid_forward` 方法



