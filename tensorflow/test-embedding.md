# embedding_lookup

```python
import tensorflow as tf

embedding = tf.get_variable("embedding", initializer=tf.ones(shape=[10, 5]))
look_uop = tf.nn.embedding_lookup(embedding, [1, 2, 3, 4])
# embedding_lookup就像是给 其它行的变量加上了stop_gradient
w1 = tf.get_variable("w", shape=[5, 1])

z = tf.matmul(look_uop, w1)

opt = tf.train.GradientDescentOptimizer(0.1)

#梯度的计算和更新依旧和之前一样，没有需要注意的
gradients = tf.gradients(z, xs=[embedding])
train = opt.apply_gradients([(gradients[0],embedding)])

#print(gradients[4])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(train))
    print(sess.run(embedding))

```
```
[[ 1.          1.          1.          1.          1.        ]
 [ 0.90580809  1.0156796   0.96294552  1.01720285  1.08395708]
 [ 0.90580809  1.0156796   0.96294552  1.01720285  1.08395708]
 [ 0.90580809  1.0156796   0.96294552  1.01720285  1.08395708]
 [ 0.90580809  1.0156796   0.96294552  1.01720285  1.08395708]
 [ 1.          1.          1.          1.          1.        ]
 [ 1.          1.          1.          1.          1.        ]
 [ 1.          1.          1.          1.          1.        ]
 [ 1.          1.          1.          1.          1.        ]
 [ 1.          1.          1.          1.          1.        ]]
```
