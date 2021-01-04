# 如何创建一个`TFRecord File`
**注意：在创建TFRecord File的时候，是不会用到tf的graph的，不会有tensor，一切都跟命令式编程一样。**


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

print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))
```
* `Example`: 一条训练样本，也称`instance`。Feature 的集合。
```python
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """

  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.

  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
```

* 写入tfrecord中
```python
with tf.python_io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
```

# 从tfrecord中解析数据
* 使用tensorflow Graph模型读数据
```python
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset
```
* 使用 python api 读数据。用来检查数据挺方便的。
```python
record_iterator = tf.python_io.tf_record_iterator(path=filename)

for string_record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(string_record)
  print(example)
```

# 参考资料
[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
[https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
[http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf](http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf)

https://github.com/tensorflow/docs/blob/r1.15/site/en/tutorials/load_data/tf_records.ipynb
