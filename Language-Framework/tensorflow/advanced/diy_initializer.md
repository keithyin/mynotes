# tensorflow 如何自定义 initializer

tensorflow中，创建变量有两个接口：

* tf.Variable()
* tf.get_variable() (如果想参数复用的话，考虑这个接口)

虽说有两个接口，但是如果创建一个新的 `Variable` 的话，`tf.get_variable` 还是会调用 `tf.Variable` 来创建变量的。



## tf.Variable

先看 tf.Variable() 接口：

```python
# class Variable(object)
def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None):
# initial_value: A `Tensor`, or Python object convertible to a `Tensor`
```

tf.Variable 的 initial_value 是个 tensor 或者是个 可转成 tensor 的 python 对象就可以。只需要搞个简单的函数来生成希望的 tensor，就可以当作一个 initializer 了。



## tf.get_variable()

再看一下 tf.get_variable() 接口：

```python
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None):
```





下面是 追`tf.get_variable()`源码得到的片段，从这可以看出，`initializer` 参数可以是：

- None
- 可调用对象 （类型对象也是可调用对象）
- Tensor 对象
- 可被 `convert_to_tensor` 的一些类型 （列表，等等）

```python
if initializer is None:
  init, initializing_from_value = self._get_default_initializer(
      name=name, shape=shape, dtype=dtype)
  if initializing_from_value:
    init_shape = None
  else:
    init_shape = var_shape
elif callable(initializer):
  init = initializer
  init_shape = var_shape
elif isinstance(initializer, ops.Tensor):
  init = array_ops.slice(initializer, var_offset, var_shape)
  # Use the dtype of the given tensor.
  dtype = init.dtype.base_dtype
  init_shape = None
else:
  init = ops.convert_to_tensor(initializer, dtype=dtype)
  init = array_ops.slice(init, var_offset, var_shape)
  init_shape = None
```



这里主要介绍如何通过 继承 `Initializer` 来创建自己的初始化器：



```python
from tensorflow.python.ops.init_ops import Initializer

class PointInitializer(Initializer):
    def __init__(self, scale, dtype=tf.float32):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return self.scale * array_ops.ones(shape, dtype)
```

继承过来的初始化器 `__init__` 方法可以自己随便写，但是 `__call__` 方法是有规范的：

* shape 参数
* dtype 参数
* partition_info 参数
* 返回值：Tensor。



## Variable 初始化到底在干什么

```python
tf.global_variables_initializer()

# global_variables_initializer 内部
return variables_initializer(global_variables())

# variables_initializer 内部
return control_flow_ops.group(*[v.initializer for v in var_list], name=name)

# v.initializer 是啥， v 是 Variable 对象
def initializer(self):
  """The initializer operation for this variable."""
  return self._initializer_op

# self._initializer_op 是啥
self._initializer_op = state_ops.assign(
            self._variable, self._initial_value,
            validate_shape=validate_shape).op  
# 就是个 assign op， 初始化的时候就是 run 了 Variable 对象的 assign op！！！
```

