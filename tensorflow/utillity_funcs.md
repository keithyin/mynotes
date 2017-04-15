# tensorflow 常用函数

## tf.add_n()
```python
tf.add_n(inputs, name=None)

Adds all input tensors element-wise.

Args:

inputs: A list of Tensor objects, each with same shape and type.
name: A name for the operation (optional).
Returns:

A Tensor of same shape and type as the elements of inputs.
```

## tf.control_dependencies()
用来控制依赖，比如说某一`op`执行完才能执行下一个`op`
对比两个函数片段：
```python
import tensorflow as tf

w1 = tf.Variable(1)

d = tf.add(w1, 1)

assi = tf.assign(w1,d)

with tf.control_dependencies([assi]):
    c = tf.add(w1, 1) #c的执行依赖assi的执行

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    assi_, c_=sess.run([assi,c]) # assi只会被执行一次,并且c是在assi后执行
    w1_ = sess.run(w1)
    print(assi_,c_,w1_)
# 输出 2,3,2
```
```python
import tensorflow as tf

w1 = tf.Variable(1)

d = tf.add(w1, 1)

assi = tf.assign(w1,d)

with tf.control_dependencies([assi]):
    c = tf.add(w1, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    c_=sess.run(c)
    w1_ = sess.run(w1)
    print(c_,w1_)
#输出 3,2
```

**可以看出，在sess.run()中如果出现了存在依赖的两个`tensor`，被依赖的`tensor`只会执计算一遍**



```python
import tensorflow as tf
state = tf.Variable(0.0,dtype=tf.float32)
one = tf.constant(1.0,dtype=tf.float32)
new_val = tf.add(state, one)
update1 = tf.assign(state, 10000)
with tf.control_dependencies([update1]):
    update2 = tf.assign(state, new_val)# 会先执行 update1
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print sess.run(update2)
#输出
# 10001
# 10001
# 10001
```

## tf.group()
```python
tf.group(*inputs, **kwargs)

Create an op that groups multiple operations.

When this op finishes, all ops in input have finished. This op has no output.

See also tuple and with_dependencies.

Args:

*inputs: Zero or more tensors to group.
**kwargs: Optional parameters to pass when constructing the NodeDef.
name: A name for this operation (optional).
Returns:

An Operation that executes all its inputs.

Raises:

ValueError: If an unknown keyword argument is provided.
```


**参考资料**
[https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph.control_dependencies](https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph.control_dependencies)
[https://www.tensorflow.org/api_docs/python/train/coordinator_and_queuerunner](https://www.tensorflow.org/api_docs/python/train/coordinator_and_queuerunner)
