# gradient

`tensorflow`中有一个计算梯度的函数`tf.gradients(ys, xs)`，要注意的是，`xs`中的`x`必须要与`ys`相关，不相关的话，会报错。
代码中定义了两个变量`w1`， `w2`， 但`res`只与`w1`相关
```python
#wrong
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

res = tf.matmul(w1, [[2],[1]])

grads = tf.gradients(res,[w1,w2])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re = sess.run(grads)
    print(re)
```
**错误信息**
TypeError: Fetch argument None has invalid type <class 'NoneType'>

```python
# right
import tensorflow as tf

w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

res = tf.matmul(w1, [[2],[1]])

grads = tf.gradients(res,[w1])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re = sess.run(grads)
    print(re)
#  [array([[2, 1]], dtype=int32)]
```
