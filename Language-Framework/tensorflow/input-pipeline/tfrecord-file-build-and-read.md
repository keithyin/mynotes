# 如何创建一个`TFRecord File`
**下面仅介绍了命令式的 tf 创建 tfrecord file 文件的方法**


tensorflow.\_\_version\_\_=1.15

抽象：Feature, Features, Example
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

# 变长数据的处理
有两种方法:
* 转成bytes存储, 可以和长度一起存储。
* 直接使用 list 存储。解析出来的是 SparseTensor
```python
# 使用 bytes 存储
def write_tf_record():
    with tf.python_io.TFRecordWriter("tfrecord.pb") as writer:
        for i in range(3):
            features = {
                "visited_city": _bytes_feature(np.array([i] * (i+1), dtype=np.int64).tostring()), # 这个 tostring 是重点。
                "length": _int64_feature(i + 1)
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example_proto.SerializeToString())

# 解析
feature_description = {
    'visited_city': tf.FixedLenFeature([], dtype=tf.string), # 依旧是一个 FixedLenFeature
    'length': tf.FixedLenFeature([], dtype=tf.int64)
}

def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    example = tf.parse_single_example(example_proto, feature_description)
    visited_city = tf.io.decode_raw(example['visited_city'], out_type=tf.int64) # decode_raw 可将 string 之前的数据还原！
    return visited_city

parsed_image_dataset = raw_image_dataset.map(_parse_function)
```

```python
def _bytes_feature(*value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[*value]))


def _float_feature(*value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[*value]))


def _int64_feature(*value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[*value]))
    
def write_tf_record():
    with tf.python_io.TFRecordWriter("tfrecord.pb") as writer:
        for i in range(3):
            features = {
                "visited_city": _int64_feature(*([i] * (i+1))),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example_proto.SerializeToString())
def parse_from_record_graph():
    record_filename = tf.placeholder(dtype=tf.string, shape=[1])
    raw_image_dataset = tf.data.TFRecordDataset(record_filename)

    # Create a dictionary describing the features.
    image_feature_description = {
        'visited_city': tf.io.VarLenFeature(dtype=tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.parse_single_example(example_proto, image_feature_description)
        # visited_city = tf.io.decode_raw(example['visited_city'], out_type=tf.int64)
        visited_city = example["visited_city"] # 因为使用 VarLenFeature解析，所以返回的是 tf.sparse.SparseTensor类型
        visited_city = tf.sparse.to_dense(visited_city) # 转成 dense 类型。
        return visited_city

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset = parsed_image_dataset.batch(1)
    iterator = parsed_image_dataset.make_initializable_iterator()

    next_val = iterator.get_next()
    #
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={record_filename: ["tfrecord.pb"]})
        while True:
            print(sess.run(next_val)) # 这边出来的是 SparsedTensor。如果不想使用 SparseTensor，可以使用第一种方式。
```

# 参考资料
[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
[https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
[http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf](http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf)

https://github.com/tensorflow/docs/blob/r1.15/site/en/tutorials/load_data/tf_records.ipynb





https://stackoverflow.com/questions/45634450/what-are-the-advantages-of-using-tf-train-sequenceexample-over-tf-train-example

https://blog.csdn.net/qq_27825451/article/details/105097430

https://blog.csdn.net/qq_27825451/article/details/105074522?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162201739616780255243098%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=162201739616780255243098&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_v29-1-105074522.nonecase&utm_term=tfrecord&spm=1018.2226.3001.4450
