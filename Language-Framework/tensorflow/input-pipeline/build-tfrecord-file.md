# 如何创建一个`TFRecord File`

tensorflow.__version__=1.15

抽象：
* `Feature`: 特征，也称`Field`字段。 对应tensorflow中的类，`tf.train.Feature`
```python
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```
* `Example`: 一条



**注意：在创建TFRecord File的时候，是不会用到tf的graph的，不会有tensor，一切都跟命令式编程一样。**

```python
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images #image的shape应该是[batch, rows, cols, depth]
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring() #转化为字节流
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()
```

## tf.train.\*\*List

* tf.train.Int64List 保存着Int64类型的列表
* tf.train.FloatList 保存着Float类型的列表
* tf.train.BytesList 保存着Bytes类型的列表

Magic attribute generated for "value" proto field
为`value`原始字段 生成的 魔术属性。
```python
int64_list = tf.train.Int64List(valule=[value])

```

## tf.train.Feature
属性：

* bytes_list
Magic attribute generated for "bytes_list" proto field.
为`bytes_list`原始字段生成的魔术属性。

* float_list
Magic attribute generated for "float_list" proto field.

* int64_list
Magic attribute generated for "int64_list" proto field.

```python
tf.train.Feature(int64_list = int64_list)
```

## tf.train.Features
属性：
* feature
Magic attribute generated for "feature" proto field.
为`feature`原始字段生成的魔术属性。

```python
tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)})
```

**看完了各种为谁生成魔术属性之后，来看看Example怎么写**
## tf.train.Example
```python
example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
# 将样本序列化后写到文件里
writer.write(example.SerializeToString())
#这里的writer就是 tf.python_io.TFRecordWriter(filename)
#写完记得close一下哦
```

一个`example` 就是一个样本。

从这看来一个原始数据需要层层包装才可以生成record文件：

```python
1. 用Int64List 包装值
2. 用Feature 包装 Int64List
3. 用Features包装 Feature
4. 用Example包装 Features
```

## 参考资料
[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
[https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
[http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf](http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf)
