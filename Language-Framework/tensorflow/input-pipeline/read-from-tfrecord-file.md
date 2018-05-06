# 从制作好的TFRecord文件中读数据

1. file_name queue
2. TFRecordReader
3. parse_single_example
4. decode_raw
5. 将 raw 还原成原数据形状

## file_name queue

使用`string_input_producer`创建一个`file_name queue`

```python
filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
```
**如果传入，num_epochs参数的话，记得用tf.local_variables_initializer().run()!!!,注意是local。原因是tf把num_epochs设置成了local variable， tf.local_variables_initializer().run()不会初始化local variable。**

## TFRecordReader
使用`TFRecordReader`从`filename_queue`中读取数据
```python
reader = tf.TFRecordReader()
key, record_string = reader.read(filename_queue)
```

## parse_single_example
定义 `parse_single_example` 来解析 `record_string`
```python
features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
              'height': tf.FixedLenFeature([], tf.int64),
              'width': tf.FixedLenFeature([], tf.int64)),
              'depth': tf.FixedLenFeature([], tf.int64),
              'label': tf.FixedLenFeature([], tf.int64),
              'image_raw': tf.FixedLenFeature([], tf.string)
#parse_single_example中features实参的key要和我们建立tfrecords时一致，但是value需要一点点改变。

```

## decode_raw
将 string 转成 uint8 类型。因为在制作`tfrecord`的时候,`image_raw`中存的是`image.tostring()`，`decode_raw`是`tostring()`的逆操作。
```python
image = tf.decode_raw(features['image_raw'], tf.uint8)
#其它的值 用 cast转成tf.int32即可
height = tf.cast(features['height'], tf.int32)
width = tf.cast(features['width'], tf.int32)
depth = tf.cast(features['depth'], tf.int32)
label = tf.cast(features['label', tf.int32])

image_shape = tf.pack([height, width, 3])
```

## 将raw数据还原

上面我们已经获取了`image_shape`，就可以reshape了
```python
img = tf.reshape(image, image_shape)
```
## 下面介绍进一步处理过程

* 对图片数据进行crop，resize 一些操作。
  [https://www.tensorflow.org/api_guides/python/image](https://www.tensorflow.org/api_guides/python/image)

## 使用tf.train.shuffle_batch

```python
example_batch, label_batch = tf.train.shuffle_batch(
      [img, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
```
```python
# 如何计算capacity
#dequeue后的所剩数据的最小值
min_after_dequeue = 10000
#queue的容量
capacity = min_after_dequeue + 3 * batch_size
```



好了，可以使用返回的`batch`进行训练了。

## 下面介绍下 Session block怎么写

1. 首先要创建一个 coord， 用于协调各线程的运行
2. 启动线程
3. 训练
4. 把线程 join 起来。
```python
with tf.Session() as sess:
  # Initialize the variables (like the epoch counter).
  tf.local_variables_initializer().run()#初始化num epoch变量
  tf.global_variables.initializer().run() # 初始化其它全局变量
  # 创建coord，用于协调各线程
  coord = tf.train.Coordinator()
  # 启动线程
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  # 训练
  try:
      while not coord.should_stop():
          # Run training steps or whatever
          sess.run(train_op)

  except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
  finally:
      # When done, ask the threads to stop.
      coord.request_stop()

  # Wait for threads to finish.
  coord.join(threads)
  sess.close()
```



## 其它

[tf.FixedLenFeature vs tf.VarLenfeature](https://stackoverflow.com/questions/41921746/tensorflow-varlenfeature-vs-fixedlenfeature)



## 参考资料

[https://www.tensorflow.org/api_guides/python/io_ops#Converting](https://www.tensorflow.org/api_guides/python/io_ops#Converting)

