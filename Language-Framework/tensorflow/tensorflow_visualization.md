# tensorflow 可视化
`tensorflow`的可视化是使用`summary`和`tensorbboard`合作完成的.

**在0.12.0版本中，下列函数的调用方法都变成了`tf.summary.scalar` 这种形式，`tf.train.SummaryWriter`改成了`tf.summary.FileWriter`, 浏览器地址变成了`127.0.1.1:6006`**
## 基本用法
首先明确一点,`summary`也是`op`.

**输出网络结构**
```python
with tf.Session() as sess:
  writer = tf.summaray.FileWriter(your_dir, sess.graph)
```
命令行运行`tensorboard --logdir=your_dir`,然后浏览器输入`127.0.1.1:6006`
这样你就可以在`tensorboard`中看到你的网络结构图了

## 可视化参数
```python

#ops
loss = ...
tf.summary.scalar("loss", loss)
merged_summary = tf.summary.merge_all()

init = tf.global_variable_initializer()
with tf.Session() as sess:
  writer = tf.summary.FileWriter(your_dir, sess.graph)
  sess.run(init)
  for i in xrange(100):
    _,summary = sess.run([train_op,merged_summary], feed_dict)
    writer.add_summary(summary, i)
```
这时,打开`tensorboard`,在`EVENTS`可以看到`loss`随着`i`的变化了


## 函数介绍
- `tf.summary.merge_all`: 将之前定义的所有`summary op`整合到一起

- `FileWriter`: 创建一个`file writer`用来向硬盘写`summary`数据,

- `tf.summary.scalar(summary_tags, Tensor/variable)`: 用于标量的 `summary`

- `tf.summary.image(tag, tensor, max_images=3, collections=None, name=None)`:tensor,必须4维,形状[batch_size, height, width, channels],`max_images`(最多只能生成3张图片的`summary`),觉着这个用在卷积中的`kernel`可视化很好用.`max_images`确定了生成的图片是[-max_images: ,height, width, channels]，还有一点就是，`TensorBord`中看到的`image summary`永远是最后一个`global step`的

- `tf.summary.histogram(tag, values, collections=None, name=None)`:values,任意形状的`tensor`,生成直方图`summary`

- `tf.summary.audio(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)`

## FileWriter
**注意**:`add_summary`仅仅是向`FileWriter`对象的缓存中存放`event data`。而向`disk`上写数据是由`FileWrite对象`控制的。下面通过`FileWriter`的构造函数来介绍这一点！！！
```python
tf.summary.FileWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)

Creates a FileWriter and an event file.
# max_queue: 在向disk写数据之前，最大能够缓存event的个数
# flush_secs: 每多少秒像disk中写数据，并清空对象缓存
```

## 注意
1. 如果使用`writer.add_summary(summary，global_step)`时没有传`global_step`参数,会使`scarlar_summary`变成一条直线。

2. 只要是在计算图上的`Summary op`，都会被`merge_all`捕捉到， 不需要考虑变量生存空间问题！
3. 如果执行一次，`disk`上没有保存`Summary`数据的话，可以尝试下`file_writer.flush()`

**参考资料**
[https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#FileWriter.__init__](https://www.tensorflow.org/api_docs/python/summary/generation_of_summaries_#FileWriter.__init__)
