# Variable
tensorflow中有两个关于variable的op，`tf.Variable()`与`tf.get_variable()`下面介绍这两个的区别

## tf.Variable与tf.get_variable()

```python
tf.Variable(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)
```
```python
tf.get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, caching_device=None, partitioner=None, validate_shape=True, custom_getter=None)
```
## 区别
1. 使用`tf.Variable`时，如果检测到命名冲突，系统会自己处理。使用`tf.get_variable()`时，系统不会处理冲突，而会报错
```python
import tensorflow as tf
w_1 = tf.Variable(3,name="w_1")
w_2 = tf.Variable(1,name="w_1")
print w_1.name
print w_2.name
#输出
#w_1:0
#w_1_1:0
```
```python
import tensorflow as tf

w_1 = tf.get_variable(name="w_1",initializer=1)
w_2 = tf.get_variable(name="w_1",initializer=2)
#错误信息
#ValueError: Variable w_1 already exists, disallowed. Did
#you mean to set reuse=True in VarScope?
```
2. 基于这两个函数的特性，当我们需要共享变量的时候，需要使用`tf.get_variable()`。在其他情况下，这两个的用法是一样的

## random Tensor
```python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

tf.random_shuffle(value, seed=None, name=None)

tf.random_crop(value, size, seed=None, name=None)

tf.multinomial(logits, num_samples, seed=None, name=None)

tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

tf.set_random_seed(seed)
```
## constant value tensor
```python
tf.zeros(shape, dtype=tf.float32, name=None)

tf.zeros_like(tensor, dtype=None, name=None)

tf.ones(shape, dtype=tf.float32, name=None)

tf.ones_like(tensor, dtype=None, name=None)

tf.fill(dims, value, name=None)

tf.constant(value, dtype=None, shape=None, name='Const')
```

**参考资料**
[https://www.tensorflow.org/api_docs/python/state_ops/variables#Variable](https://www.tensorflow.org/api_docs/python/state_ops/variables#Variable)
[https://www.tensorflow.org/api_docs/python/state_ops/sharing_variables#get_variable](https://www.tensorflow.org/api_docs/python/state_ops/sharing_variables#get_variable)
[https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op/](https://www.tensorflow.org/versions/r0.10/api_docs/python/constant_op/)
