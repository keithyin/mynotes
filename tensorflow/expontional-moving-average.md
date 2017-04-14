# ExponentialMovingAverage
Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly.
`tensorflow` 官网上对于这个方法功能的介绍。`GradientDescent` 和 `Momentum` 方式的训练 都能够从 `ExponentialMovingAverage` 方法中获益。

**什么是MovingAverage?**
假设我们与一串时间序列 $$\{a_1, a_2, a_3, ..., a_{t-1}, a_t, ...\}$$,那么，这串时间序列的 `MovingAverage` 就是：
$$
mv_t = decay*mv_{t-1}+(1-decay)* a_t
$$
这是一个递归表达式。
**如何理解这个式子呢？**
他就像一个滑动窗口，$mv_t$ 的值只和这个窗口内的 $a_i$ 有关， 为什么这么说呢？将递归式拆开 :
$$
\begin{aligned}
mv_t &= (1-decay)* a_t+decay*mv_{t-1} \\
mv_{t-1} &= (1-decay)* a_{t-1}+decay*mv_{t-2} \\
mv_{t-2} &= (1-decay)* a_{t-2}+decay*mv_{t-3}\\
&...
\end{aligned}
$$
得到：
$$
mv_t = \sum_{i=1}^tdecay^{t-i}* (1-decay)* a_i
$$
当 $t-i>C$， $C$ 为某足够大的数时 $$decay^{t-i}* (1-decay)* a_i \approx 0$$
, 所以:
$$
mv_t\approx \sum_{i=t-C}^tdecay^{t-i}* (1-decay)* a_i
$$。即， $mv_t$ 的值只和 $\{a_{t-C},...,a_t\}$ 有关。

## tensorflow 中的 ExponentialMovingAverage
这时，再看官方文档中的公式:
$$
\text{shadowVariable} = decay * \text{shadowVariable} + (1 - decay) * variable
$$,就知道各代表什么意思了。
```python
import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

with tf.control_dependencies([update]):
    ema_op = ema.apply([w])
ema_val = ema.average(w)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run(ema_val))
# 创建一个时间序列 1 2 3 4
#输出：
#1.1      =0.9*1 + 0.1*2
#1.29     =0.9*1.1+0.1*3
#1.561    =0.9*1.29+0.1*4
```
你可能会奇怪，明明 只执行三次循环， 为什么产生了 4 个数？
这是因为，当程序执行到 `ema_op = ema.apply([w])` 的时候，如果 `w` 是 `Variable`， 那么将会用 `w` 的初始值初始化 `ema` 中关于 `w` 的 `ema_value`，所以 $emaVal_0=1.0$。如果 `w` 是 `Tensor`的话，将会用 `0.0` 初始化。

官网中的示例：
```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)
    # run这个op获取当前时刻 ema_value
    get_var0_average_op = ema.average(var0)
```

## 使用 ExponentialMovingAveraged parameters
假设我们使用了`ExponentialMovingAverage`方法训练了神经网络， 在`test`阶段，如何使用 `ExponentialMovingAveraged parameters`呢？ 官网也给出了答案
**方法一：**
```python
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```
**方法二：**
```python
#Returns a map of names to Variables to restore.
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
```
# 参考资料
[https://www.tensorflow.org/versions/master/api_docs/python/train/moving_averages](https://www.tensorflow.org/versions/master/api_docs/python/train/moving_averages)
