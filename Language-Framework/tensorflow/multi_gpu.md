# Distribuited tensorflow
## Multiple GPU
![Multiple GPUs](https://www.tensorflow.org/versions/r0.10/images/Parallelism.png)
### 如何设置训练系统
(1)每个GPU上都会有model的副本
(2)对模型的参数进行同步更新

### 抽象名词
- 计算单个副本`inference`和 `gradients` 的函数称之为`tower`,使用tf.name_scope()为tower中的每个op_name加上前缀
- 使用`tf.device('/gpu:0')` 来指定tower中op的运算设备
### 框架:
```python
with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)
    tower_grads=[]
    for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                #这里定义你的模型
                #ops,variables

                #损失函数
                loss = yourloss
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Calculate the gradients for the batch of data on this tower.
                grads = opt.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads)
```
[源码地址](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)
