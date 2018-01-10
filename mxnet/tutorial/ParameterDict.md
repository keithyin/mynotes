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

