# learning rate decay
在训练神经网络的时候，通常在训练刚开始的时候使用较大的`learning rate`， 随着训练的进行，我们会慢慢的减小`learning rate`。对于这种常用的训练策略，`tensorflow` 也提供了相应的`API`让我们可以更简单的将这个方法应用到我们训练网络的过程中。

**接口**
`tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)`
参数:
`learning_rate` : 初始的`learning rate`
`global_step` : 全局的step，与 `decay_step` 和 `decay_rate`一起决定了 `learning rate`的变化。

更新公式：
```python
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
```
用法：
```python
import tensorflow as tf

global_step = tf.Variable(0)

initial_learning_rate = 0.1 #初始学习率

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=10,decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(learning_rate))
    for i in range(10):
        _, rate = sess.run([add_global, learning_rate])
        print(rate)

```

## 参考资料
[https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)
