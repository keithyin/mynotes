# tensorflow

`tf.control_dependencies()`设计是用来控制计算流图的，给图中的某些计算指定顺序。比如：我们想要获取参数更新后的值，那么我们可以这么组织我们的代码。
```python

opt = tf.train.Optimizer().minize(loss)

with tf.control_dependencies([opt]):
  updated_weight = tf.identity(weight)

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  sess.run(updated_weight, feed_dict={...}) # 这样每次得到的都是更新后的weight
```
关于tf.control_dependencies的具体用法，請移步官网[https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies](https://www.tensorflow.org/api_docs/python/tf/Graph#control_dependencies),总结一句话就是，在执行某些`op tensor`之前，某些`op tensor`得首先被运行。

## 下面说明两种 control_dependencies 不 work 的情况

下面有两种情况，control_dependencies不work，其实并不是它真的不work，而是我们的使用方法有问题。

**第一种情况:**

```python
import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    ema_val = ema.average(update)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        print(sess.run([ema_val]))
```
也许你会觉得，在我们 `sess.run([ema_val])`， `ema_op` 都会被先执行，然后再计算`ema_val`，实际情况并不是这样，为什么？
有兴趣的可以看一下源码，就会发现 `ema.average(update)` 不是一个 `op`，它只是从`ema`对象的一个字典中取出键对应的 `tensor` 而已，然后赋值给`ema_val`。这个 `tensor`是由一个在 `tf.control_dependencies([ema_op])` 外部的一个 `op` 计算得来的，所以 `control_dependencies`会失效。解决方法也很简单，看代码：

```python
import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    ema_val = tf.identity(ema.average(update)) #一个identity搞定

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        print(sess.run([ema_val]))
```

**第二种情况：** 这个情况一般不会碰到，这是我在测试 `control_dependencies` 发现的

```python
import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

ema_op = ema.apply([update])
with tf.control_dependencies([ema_op]):
    w1 = tf.Variable(2.0)
    ema_val = ema.average(update)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        print(sess.run([ema_val, w1]))
```
这种情况下，`control_dependencies`也不 work。读取 `w1` 的值并不会触发 `ema_op`， 原因不大清楚。。。猜测是`read op`，并不会被`control_dependencies`
