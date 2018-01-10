# ParameterDict

* `Block` 中的一个属性： 用于存放参数。

**用于**

* 保存参数
* 初始化参数
* 加载预训练的参数



## 保存参数

```python
class ParameterDict(object):
    def __init__(self, prefix='', shared=None):
        self._prefix = prefix
        # 参数保存在 OrderedDict 中
        self._params = OrderedDict()
        self._shared = shared
        
```



## 初始化参数

```python
def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        """
        初始化所有这个 ParameterDict 管理的参数
        API. It has no effect when using :py:class:`Symbol` API.

        Parameters
        ----------
        init : Initializer
            Global default Initializer to be used when `Parameter.init` is None.
            Otherwise, :py:meth:`Parameter.init` takes precedence.
        ctx : Context or list of Context
            Keeps a copy of Parameters on one or many context(s).
        force_reinit : bool, default False
            Whether to force re-initialization if parameter is already initialized.
        """
    if verbose:
        init.set_verbosity(verbose=verbose)
    for _, v in self.items():
        v.initialize(None, ctx, init, force_reinit=force_reinit)
```

* 到后来由 `Parameter` 自己来负责自己的初始化。

**MXNET的参数初始化逻辑是**：

* 对每一个参数应用 `Initializer`， 然后根据 参数的名字来确定自己该怎么搞。



```python
# Parameter 成员方法
def initialize(self, init=None, ctx=None, default_init=initializer.Uniform(),
               force_reinit=False):
    """Initializes parameter and gradient arrays. Only used for :py:class:`NDArray` API.

    Parameters
    ----------
    init : Initializer
        The initializer to use. Overrides :py:meth:`Parameter.init` and default_init.
    ctx : Context or list of Context, defaults to :py:meth:`context.current_context()`.
        Initialize Parameter on given context. If ctx is a list of Context, a
        copy will be made for each context.

        .. note::
            Copies are independent arrays. User is responsible for keeping
            their values consistent when updating.
            Normally :py:class:`gluon.Trainer` does this for you.

    default_init : Initializer
        Default initializer is used when both :py:func:`init`
        and :py:meth:`Parameter.init` are ``None``.
    force_reinit : bool, default False
        Whether to force re-initialization if parameter is already initialized.

    Examples
    --------
    >>> weight = mx.gluon.Parameter('weight', shape=(2, 2))
    >>> weight.initialize(ctx=mx.cpu(0))
    >>> weight.data()
    [[-0.01068833  0.01729892]
     [ 0.02042518 -0.01618656]]
    <NDArray 2x2 @cpu(0)>
    >>> weight.grad()
    [[ 0.  0.]
     [ 0.  0.]]
    <NDArray 2x2 @cpu(0)>
    >>> weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])
    >>> weight.data(mx.gpu(0))
    [[-0.00873779 -0.02834515]
     [ 0.05484822 -0.06206018]]
    <NDArray 2x2 @gpu(0)>
    >>> weight.data(mx.gpu(1))
    [[-0.00873779 -0.02834515]
     [ 0.05484822 -0.06206018]]
    <NDArray 2x2 @gpu(1)>
    """
    if self._data is not None and not force_reinit:
        warnings.warn("Parameter %s is already initialized, ignoring. " \
                      "Set force_reinit=True to re-initialize."%self.name)
        return
    self._data = self._grad = None

    if ctx is None:
        ctx = [context.current_context()]
    if isinstance(ctx, Context):
        ctx = [ctx]
    if init is None:
        init = default_init if self.init is None else self.init
    # 如果初始化信息不足，就延时初始化
    # 估计在第一次前向的时候， Parameter 的 shape 会 修改。
    # 然后就可以 初始化了。
    if not self.shape or np.prod(self.shape) <= 0:
        if self._allow_deferred_init:
            self._deferred_init = (init, ctx, default_init, None)
            return
        raise ValueError("Cannot initialize Parameter %s because it has " \
                         "invalid shape: %s."%(self.name, str(self.shape)))

    self._deferred_init = (init, ctx, default_init, None)
    self._finish_deferred_init()
```



```python
# Parameter 成员方法， 执行初始化
def _finish_deferred_init(self):
    """Finishes deferred initialization."""
    if not self._deferred_init:
        return
    init, ctx, default_init, data = self._deferred_init
    self._deferred_init = ()
    assert self.shape is not None and np.prod(self.shape) > 0, \
        "Cannot initialize Parameter %s because it has " \
        "invalid shape: %s. Please specify in_units, " \
        "in_channels, etc for `Block`s." % (
            self.name, str(self.shape))

    with autograd.pause():
        if data is None:
            # shape 已经有了，所以可以操作。
            data = ndarray.zeros(shape=self.shape, dtype=self.dtype,
                                 ctx=context.cpu())
            # 
            initializer.create(default_init)(
                initializer.InitDesc(self.name, {'__init__': init}), data)

        self._init_impl(data, ctx)

```







**initializer.Initializer**

```python
class Initializer(object):
    """The base class of an initializer."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._verbose = False
        self._print_func = None
    def __call__(self, desc, arr):
        """Initialize an array

        Parameters
        ----------
        desc : InitDesc
            Initialization pattern descriptor.

        arr : NDArray
            The array to be initialized.
        """
        if not isinstance(desc, InitDesc):
            self._legacy_init(desc, arr)
            return

        if desc.global_init is None:
            desc.global_init = self
        init = desc.attrs.get('__init__', "")

        if init:
            # when calling Variable initializer
            create(init)._init_weight(desc, arr)
            self._verbose_print(desc, init, arr)
        else:
            # register nnvm::FSetInputVariableAttrs in the backend for new patterns
            # don't add new cases here.
            if desc.endswith('weight'):
                self._init_weight(desc, arr)
                self._verbose_print(desc, 'weight', arr)
            elif desc.endswith('bias'):
                self._init_bias(desc, arr)
                self._verbose_print(desc, 'bias', arr)
            elif desc.endswith('gamma'):
                self._init_gamma(desc, arr)
                self._verbose_print(desc, 'gamma', arr)
            elif desc.endswith('beta'):
                self._init_beta(desc, arr)
                self._verbose_print(desc, 'beta', arr)
            else:
                self._init_default(desc, arr)
    def _init_bilinear(self, _, arr):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i / shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        arr[:] = weight.reshape(shape)

    def _init_loc_bias(self, _, arr):
        shape = arr.shape
        assert(shape[0] == 6)
        arr[:] = np.array([1.0, 0, 0, 0, 1.0, 0])

    def _init_zero(self, _, arr):
        arr[:] = 0.0

    def _init_one(self, _, arr):
        arr[:] = 1.0

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")
```

