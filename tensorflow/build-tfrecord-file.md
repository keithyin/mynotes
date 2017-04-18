# 如何创建一个`TFRecord File`


```python
def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
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
```
##

## tf.train.Example

## tf.train.\*\*List
* tf.train.Int64List
* tf.train.FloatList
* tf.train.BytesList

## tf.train.Feature
属性：

* bytes_list
Magic attribute generated for "bytes_list" proto field.

* float_list
Magic attribute generated for "float_list" proto field.

* int64_list
Magic attribute generated for "int64_list" proto field.

## tf.train.Features
属性：
* feature
Magic attribute generated for "feature" proto field.

## 参考资料
[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
