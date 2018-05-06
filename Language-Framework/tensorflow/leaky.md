# tensorflow leaky relu
在`tensorflow 0.12.0`及之前，都没有内置的`leaky relu`函数，那么我们如何实现`leaky relu`函数呢？
## 方法1
```python
def relu(x, alpha=0., max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    x -= tf.constant(alpha, dtype=_FLOATX) * negative_part
    return x
```
## 方法2
```python
x = tf.maximum(alpha*x,x)
```
这两种方法，在`BP`的时候，梯度都会被正确的计算的
**参考资料**
[https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE)
