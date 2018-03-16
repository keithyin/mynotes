# 通过 tf.data API 读取数据

`tf.data` API 在 `tensorflow1.4` 版本加入了 `tensorflow`, 提供更加简便的 数据读取方法. 在没有`tf.data` 之前, 需要使用 `queue_runner(), Coordinate()` 这些方法, 是非常麻烦的.



**使用 tf.data API 的基本流程为**

```python
dataset = tf.data.Dataset(...)
dataset = dataset.map(_map_func) # 解析函数
dataset = dataset.shuffle(...) # 是否对数据集进行 shuffle, 不是完全 shuffle (可选)
dataset = dataset.batch(...) # 设置 batch_size 
dataset = dataset.repeat(...) # 重复多少次数据集 (可选)
iterator = dataset.make_one_shot_iterator() # 通过dataset 创建一个 迭代器
next_batch = iterator.get_next() # 获取下一个 batch

# 
session.run(next_batch) #来获取每个 batch 就可以了.
```



## tf.data.Dataset

**构建 Dataset 的方法**

```python
# 使用内存中的数据构建 Dataset
tf.data.Dataset.from_tensors()
tf.data.Dataset.from_tensor_slices()

# 使用 TFRecord 构建 Dataset
tf.data.TFRecordDataset(filenames)
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



## 迭代器

`tf.data API` 提供了多种迭代器供选择

> 多种迭代器
>
> * one-shot: 一次性迭代器, 用过就完
> * initializable: 可初始化的迭代器, 可多次使用
> * reinitializable: ...
> * feedable: ...

```python
# one-shot, 不支持参数化, 不能重新初始化, repeat, batch, shuffle 都可以用
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

```python
# initializable, 可初始化的迭代器, 支持参数化.
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
    demo = Demo()
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
    demo = Demo()
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





## 参考资料

[https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)