# queues
**Queues are a powerful mechanism for asynchronous computation using TensorFlow**
这是官方教程中对于`threading and queues`的介绍的第一句话，足以说明研究一下这东西怎么是很有价值的。

`queue`即队列，具有先进先出的性质。在`tensorflow`中，用`q = tf.FIFOQueue(...)`来创建。

## tensorflow 中的 queue
`tensorflow`中所有都是`node`(节点)。`tensorflow`中的`node`可以分成两类，一类是`op`，另一类就像`variable`，可以保存状态的。`queue`就属于保存状态的。回忆`tf.Variable`,它通过`tf.assign()`来改变自身的状态。而对于`queue`来说，它是通过`q.enqueue()`和`q.dequeue()`来改变自身状态。

## tf.Coordinator
`tf.Coordinator` 用于保证 多线程的时候，所有线程一起停止
如何使用 `tf.Coordinator`?

## tf.QueueRunner
`The QueueRunner class is used to create a number of threads cooperating to enqueue tensors in the same queue.`
`The QueueRunner class creates a number of threads that repeatedly run an enqueue op. These threads can use a coordinator to stop together.` 



## 如何使用 tf 中 Queue 来保存 更复杂的数据

当查看 tensorflow 中提供的 `Queue` 的类的时候，会发现有个 `dtypes` 参数，这个参数代表 `Queue` 中元素是什么类型的，如果 `dtypes=[tf.string, tf.int64]` ，这个表示，`Queue` 中每个元素是 `(string, int)` 。

```python
import tensorflow as tf

queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.int64])

# enqueue_many 的写法，两个元素放在连个列表里。
en_m = queue.enqueue_many([['hello', 'world'], [1, 2]])

# enqueue 的写法
en = queue.enqueue(['hello', 1])
deq = queue.dequeue()

with tf.Session() as sess:
    sess.run(en_m)
    print(sess.run(deq))
```

