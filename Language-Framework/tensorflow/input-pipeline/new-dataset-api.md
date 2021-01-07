# 通过 tf.data API 读取数据

[https://www.tensorflow.org/guide/datasets](https://www.tensorflow.org/guide/datasets)



`tf.data` API 在 `tensorflow1.4` 版本加入了 `tensorflow`, 提供更加简便的 数据读取方法. 在没有`tf.data` 之前, 需要使用 `queue_runner(), Coordinate()` 这些方法, 是非常麻烦的.



**使用 tf.data API 的基本流程为**

* 设置 Dataset
* 构建 iterator

```python
dataset = tf.data.Dataset(...)
dataset = dataset.map(_map_func) # 解析函数, 对Dataset 中的每个 sample 做此操作
dataset = dataset.shuffle(...) # 是否对数据集进行 shuffle, 不是完全 shuffle (可选)
dataset = dataset.batch(...) # 设置 batch_size 
dataset = dataset.repeat(...) # 重复多少次数据集 (可选)
iterator = dataset.make_one_shot_iterator() # 通过dataset 创建一个 迭代器
next_batch = iterator.get_next() # 获取下一个 batch

# 
session.run(next_batch) #来获取每个 batch 就可以了.
```



# tf.data.Dataset

**构建 Dataset 的方法**

```python
# 使用内存中的数据构建 Dataset
tf.data.Dataset.from_tensors()
tf.data.Dataset.from_tensor_slices()

# 使用 TFRecord 构建 Dataset
tf.data.TFRecordDataset(filenames)

# 使用 generator 构建 Dataset
tf.data.from_generator()
```

```python
# 函数对象均可
def GenerateData():
    datas = []
    tmp = []
    for i in range(100):
        tmp.append(i)
        datas.append(copy.deepcopy(tmp))
    for idx in range(101):
        tmp = datas[idx]
        yield (tmp, )

class GenerateData2(object):
    def __init__(self):
        self.datas = []
        self.cur_idx = 0
        self._generate_data()

    def _generate_data(self):
        tmp = []
        for i in range(100):
            tmp.append(i)
            self.datas.append(copy.deepcopy(tmp))

    def __call__(self):
        for i in range(100):
            tmp = self.datas[i]
            yield (tmp, )

if __name__ == "__main__":
    dataset = tf.data.Dataset.from_generator(
        GenerateData, (tf.int64, ), output_shapes=(tf.TensorShape([None]), ))
    dataset2 =  tf.data.Dataset.from_generator(
        GenerateData2(), (tf.int64, ), 
        output_shapes=(tf.TensorShape([None]), ))
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()
    with tf.Session() as sess:
        for i in range(101):
            print(sess.run(next_data))
    print("hello world")
```





**Dataset.map**

> 定义对每个样本的操作
>
> * 解码
> * 数据增强
> * ......



```python
Dataset.map(func) 
# func 定义对每个样本的操作

# 如果是 TFRecord, func 的形参是 example_proto
def func(example_proto):
    pass
```

# 变长数据的操作
有两种方式处理变长数据
1. parse_single_example 的时候 to_dense(), batch 的时候使用 padded_batch
2. parse_single_example 的时候不做操作，batch 的时候 使用  batch(), 模型中使用的时候 `tf.sparse.to_dense()`。 这个结果与上述一致。
```python
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
        visited_city = example["visited_city"]
        visited_city = tf.sparse.to_dense(visited_city)
        return {"visited_city": visited_city}, tf.shape(visited_city)[0]

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    # 这里一定要注意 padded_shapes 的值，其格式一定要与前一步dataset返回的结果格式匹配上！！！！特别是 tuple 和 list不要混。
    # (None, )表示对于原始数据的第一维进行pad，如果不想pad的话 (1, ) 直接写明tensor的shape即可。如果有存在标量使用 () 即可。
    parsed_image_dataset = parsed_image_dataset.padded_batch(2, padded_shapes=({'visited_city': (None,)}, ()))
    iterator = parsed_image_dataset.make_initializable_iterator()

    next_val = iterator.get_next()
    #
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={record_filename: ["tfrecord.pb"]})
        while True:
            print(sess.run(next_val))
```

```python
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
        visited_city = example["visited_city"]
        return {"visited_city": visited_city}, tf.shape(visited_city)[0]

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
   
    parsed_image_dataset = parsed_image_dataset.batch(2)
    iterator = parsed_image_dataset.make_initializable_iterator()

    next_val = iterator.get_next()
    next_val[0]['visited_city'] = tf.sparse.to_dense(next_val[0]['visited_city'])
    
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={record_filename: ["tfrecord.pb"]})
        while True:
            print(sess.run(next_val))
```

# 迭代器

`tf.data API` 提供了多种迭代器供选择

> 多种迭代器
>
> * one-shot: 一次性迭代器, 用过就完
> * initializable: 可初始化的迭代器, 可多次使用
>   * 可以使用 placeholder 参数化 Dataset 的某些定义
> * reinitializable: 
>   * 可以用不同的 `Dataset 对象` 初始化之，
>   * 可以使用多个 Dataset 的 iterator
> * feedable: 
>   * 一个可以切换不同的 iterator 的 iterator
>   * 可以使用多个 iterator 的 iterator



**make_one_shot_iterator**

```python
# one-shot, 不支持参数化, 不能重新初始化, repeat, batch, shuffle 可以用
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```



**make_initializable_iterator**

```python
# initializable, 可初始化的迭代器, 支持参数化 Dataset
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```



**reinitializable**

* 使用结构定义，一个多个 `Dataset` 可以共用一个`Iterator` 

```python
# 用结构定义，而不是用 Dataset 进行定义，所以可以应用到多个 Dataset
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```



**feedable iterator**

* 可以在使用中切换 `iterator` ，通过 `feed_dict`
* 切换时不需要初始化 `iterator`，可以保存 `iterator ` 的状态

```python
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:
  # Run 200 steps using the training dataset. Note that the training dataset is
  # infinite, and we resume from where we left off in the previous `while` loop
  # iteration.
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # Run one pass over the validation dataset.
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})

```



## Effective tf.data

[https://www.tensorflow.org/guide/performance/datasets](https://www.tensorflow.org/guide/performance/datasets)



### Parallelize Data Extraction

-----

```python
# 没有用 parallelize data extraction 的
files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
dataset = files.interleave(tf.data.TFRecordDataset)

# 使用 parallelize data extraction
dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
```





### Parallelize Data Transformation

----

* 在 `dataset pipeline` 的最后使用 `prefetch(1)` 

```python
dataset = dataset.batch(batch_size=FLAGS.batch_size)
dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
return dataset
```

* 给 `map` 指定 `num_parallel_calls` 参数，让 `map` 并行起来

```python
dataset = dataset.map(map_func=parse_fn, 
                      num_parallel_calls=FLAGS.num_parallel_calls)
```

* 如果 batch 的数量比较大的话，那就使用 融合的 api

```python
#  batch大 不建议使用
dataset = dataset.map(map_func=parse_fn, 
                      num_parallel_calls=FLAGS.num_parallel_calls)
dataset = dataset.batch(batch_size=FLAGS.batch_size)


# batch 大建议使用
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=parse_fn, batch_size=FLAGS.batch_size))
```





## 如何写出更高效的 input pipeline

* `Map and Batch`: 如果 `map` 执行的操作非常少，那就可以先 `Batch`，然后再 `Map` 
* `Repeat and Shuffle`： 
  * `repeat before shuffle` : provides better performance
  * `shuffle before repeat` ：provides stronger ordering guarantees 
  * 使用 `contrib.data.shuffle_and_repeat` 效果会更好哦



## 如何将 tf.data API 封装起来使得更 pythonic

```python
class DataIter():
    def __init__(self, filenames, batch_size=100):
        """创建输入流水线计算图, 可以加上测试集
        dataset = tf.data.Dataset(...)
		dataset = dataset.map(_map_func) # 解析函数
		dataset = dataset.shuffle(...) # 是否对数据集进行 shuffle, 不是完全 shuffle (可选)
		dataset = dataset.batch(...) # 设置 batch_size 
		dataset = dataset.repeat(...) # 重复多少次数据集 (可选)
		self.iterator = dataset.make_one_shot_iterator() # 通过dataset 创建一个 迭代器
		self.next_batch = iterator.get_next() # 获取下一个 batch
		
		如果有测试集:可以设置一个 flag
		self._train = True, 然后通过 self.train(), self.eval() 切换状态
        """
        
        """实例一个 session
        self.session = tf.Session
        """
        pass
    def init_input_pipeline(self):
        """如果使用了 dataset.make_initializable_iterator
        使用这个方法对输入流水线进行 初始化
        self.session.run(self.iterator.initializer,
        			feed_dict={features_placeholder: features,
                               labels_placeholder: labels})
        """
        pass
    def __next__(self):
        """
        根据 self._train 返回 self.session.run(self.next_batch)
        """
        pass
    
```



## 例子

```python
import tensorflow as tf
class DataIter():
    def __init__(self):
        # prepare train  dataset
        train_dataset = tf.data.Dataset.range(97)
        train_dataset = train_dataset.batch(10)
        self.train_data_iter = train_dataset.make_initializable_iterator()
        self.next_train_batch = self.train_data_iter.get_next()

        # prepare val dataset
        val_dataset = tf.data.Dataset.range(32)
        val_dataset = val_dataset.batch(10)
        self.val_data_iter = val_dataset.make_initializable_iterator()
        self.next_val_batch = self.val_data_iter.get_next()
		
        # lock the tensorflow graph
        tf.get_default_graph().finalize() 
        self.session = tf.Session()
		
        # train flag
        self._train = True

    def __iter__(self):
        return self
    
	# 这里是将 train 和 val 的初始化分开,
    # 也可以将他们放到一起, 然后通过 iterator.train(), iterator.eval() 
    # 来切换返回的值
    def init_train_pipeline(self):
        self.train()
        self.session.run(self.train_data_iter.initializer)

    def init_val_pipeline(self):
        self.eval()
        self.session.run(self.val_data_iter.initializer)

    def __next__(self):
        try:
            if self._train:
                return self.session.run(self.next_train_batch)
            else:
                return self.session.run(self.next_val_batch)
        except tf.errors.OutOfRangeError:
            raise StopIteration("out of range")

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

if __name__ == '__main__':
    demo = DataIter()
    for i in range(10):
        demo.init_train_pipeline()
        for v in demo:
            print(v)
        print("--------------------------")
        demo.init_val_pipeline()
        for v in demo:
            print(v)
        print("**************************")
```



**第二种方式:  更灵活的解决方案**

```python
import tensorflow as tf
class DataIter():
    def __init__(self):
        """
        和上面一样
        """
        pass

    def __iter__(self):
        return self
	
    def init_pipeline(self):
        self.session.run(self.train_data_iter.initializer)
        self.session.run(self.val_data_iter.initializer)
        
    def __next__(self):
        try:
            if self._train:
                return self.session.run(self.next_train_batch)
            else:
                return self.session.run(self.next_val_batch)
        except tf.errors.OutOfRangeError:
            raise StopIteration("out of range")

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

if __name__ == '__main__':
    demo = DataIter()
    for i in range(10):
        demo.init_pipeline()
        demo.train()
        for v in demo:
            print(v)
        print("--------------------------")
        demo.eval()
        for v in demo:
            print(v)
        print("**************************")
```

```python
# in action
import tensorflow as tf
import os


class DataIter():
    def __init__(self):
        # prepare train  dataset
        self.train_files = tf.placeholder(dtype=tf.string, shape=[None])
        train_dataset = tf.data.TFRecordDataset(self.train_files)
        train_dataset = train_dataset.map(DataIter.map_func)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.batch(100)
        self.train_data_iter = train_dataset.make_initializable_iterator()
        self.next_train_batch = self.train_data_iter.get_next()

        # prepare val dataset
        self.val_files = tf.placeholder(tf.string, shape=[None])
        val_dataset = tf.data.TFRecordDataset(self.val_files)
        val_dataset = val_dataset.map(DataIter.map_func)
        val_dataset = val_dataset.batch(100)
        self.val_data_iter = val_dataset.make_initializable_iterator()
        self.next_val_batch = self.val_data_iter.get_next()

        # lock the tensorflow graph
        tf.get_default_graph().finalize()
        self.session = tf.Session()

        # train flag
        self._train = True

    def __iter__(self):
        return self

    def init_pipeline(self, filepathes, holdout_id=1):
        trains = [file for i, file in enumerate(filepathes, start=1)
                  if i != holdout_id]
        vals = [file for i, file in enumerate(filepathes, start=1)
                if i == holdout_id]
        self.session.run(self.train_data_iter.initializer, feed_dict={
            self.train_files: trains
        })
        self.session.run(self.val_data_iter.initializer, feed_dict={
            self.val_files: vals
        })

    def __next__(self):
        try:
            if self._train:
                return self.session.run(self.next_train_batch)
            else:
                return self.session.run(self.next_val_batch)
        except tf.errors.OutOfRangeError:
            raise StopIteration("out of range")

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    @staticmethod
    def map_func(example):
        features = {"label": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                    "audio_raw": tf.FixedLenFeature(shape=[], dtype=tf.string)}
        example = tf.parse_single_example(example, features)
        label = example["label"]
        audio = tf.reshape(tf.decode_raw(example["audio_raw"], out_type=tf.float32),
                           shape=[2, 60, 101])
        return audio, label


if __name__ == '__main__':

    root = "/media/fanyang/workspace/DataSet/ESC-50-master/tfrecords"
    files = sorted(os.listdir(root))
    filepaths = [os.path.join(root, file) for file in files]

    data_iter = DataIter()
    for i in range(10):
        data_iter.init_pipeline(filepaths)
        data_iter.train()
        for batch in data_iter:
            print(batch)
        print("--------------------------")
        data_iter.eval()
        for batch in data_iter:
            print(batch)
        print("**************************")

```



# 使用input-pipeline 的最佳实践

* tensorflow 的输入流水线包含
  * 数据处理 和 模型训练 的流水线
  * 数据处理 内部 的流水线

```python
# 数据处理 与 模型训练的 流水线.   这个 prefetch 操作是有个 background 线程 来执行.
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```



```python
# 数据处理内部的 流水线, 并行化 map 操作 (data transformation)
dataset = dataset.map(map_func=parse_fn) # 串行
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) # 流水线
```



```python
# 并行化数据解析, 如果数据在 remote 或者需要 解码
dataset = tf.data.TFRecordDataset(files)
# 换成
dataset = files.interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_reads,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
```



# 参考资料

[https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)

https://www.tensorflow.org/guide/data_performance
