# caffe2: 如何创建自己的数据集

`caffe2` 使用 二进制 `DB` 格式来保存用于训练模型的代码. `DB` 只是 `key-value` 的另一个高大上的名字而已, `key`通常是随机的, 这样可以保证 `batches` 大概是 `iid` 的. 至于 `value`, 它保存了训练数据的 序列化 `string`. 所以, 保存的 `DB` 语义上大概是这样:

```
key1 value1 key2 value2 key3 value3 …
```

对于 `DB` 来说, 它把 `key` 和 `value` 都当作是 `string`, 但是, 也许你想要结构化的内容. 一个方法是使用 `TensorProtos  protocol buffer`: 本质上,它是吧 `tensor` 的值,类型(int,float...), 形状(shape) 包起来. 然后,就可以使用 `TensorProtosDBInput` `operator` 把它加载进来了.

在这里,会给出一个创建自己的数据集的例子. 为此, 我们将使用 `UCI Iris` 数据集 - 用与分类花的数据集,有3类.数据可以在此[下载](https://archive.ics.uci.edu/ml/datasets/Iris).

```python
import urllib2 # for downloading the dataset from the web.
import numpy as np
from matplotlib import pyplot
from StringIO import StringIO
from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2

f = urllib2.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
raw_data = f.read()
print('Raw data looks like this:')
print(raw_data[:120] + '...')

```
```
Raw data looks like this:
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,...
```

```python
# load the features to a feature matrix.
features = np.loadtxt(StringIO(raw_data), dtype=np.float32, delimiter=',', usecols=(0, 1, 2, 3))
# load the labels to a feature matrix
label_converter = lambda s : {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}[s]
labels = np.loadtxt(StringIO(raw_data), dtype=np.int, delimiter=',', usecols=(4,), converters={4: label_converter})
```
在开始训练之前, 将数据集分成训练集和测试集是一个很好的方案. 在这种情况下,让我们随即打乱数据, 然后使用前100个样本用于训练,剩下的样本用来测试.当然, 你也可以使用交叉验证.

```python
random_index = np.random.permutation(150)
features = features[random_index]
labels = labels[random_index]

train_features = features[:100]
train_labels = labels[:100]
test_features = features[100:]
test_labels = labels[100:]
```
```python
# Let's plot the first two features together with the label.
# Remember, while we are plotting the testing feature distribution
# here too, you might not be supposed to do so in real research,
# because one should not peek into the testing data.
legend = ['rx', 'b+', 'go']
pyplot.title("Training data distribution, feature 0 and 1")
for i in range(3):
    pyplot.plot(train_features[train_labels==i, 0], train_features[train_labels==i, 1], legend[i])
pyplot.figure()
pyplot.title("Testing data distribution, feature 0 and 1")
for i in range(3):
    pyplot.plot(test_features[test_labels==i, 0], test_features[test_labels==i, 1], legend[i])
pyplot.show()
```

现在, 按照之前所说, 我们把数据处理成 `caffe2 DB`. 在这个`DB`中, 我们会使用 `train_xxx` 作为`key`, 使用`TensorProtos` 对象 为每个数据点存储两个 `tensors`:一个作为 `feature`, 一个作为`label`. 我们将会使用 `caffe2` 的 `python` `DB` 接口:

```python
# First, let's see how one can construct a TensorProtos protocol buffer from numpy arrays.
feature_and_label = caffe2_pb2.TensorProtos()
feature_and_label.protos.extend([
    utils.NumpyArrayToCaffe2Tensor(features[0]),
    utils.NumpyArrayToCaffe2Tensor(labels[0])])
print('This is what the tensor proto looks like for a feature and its label:')
print(str(feature_and_label))
print('This is the compact string that gets written into the db:')
print(feature_and_label.SerializeToString())
```

```
This is what the tensor proto looks like for a feature and its label:
protos {
  dims: 4
  data_type: FLOAT
  float_data: 4.40000009537
  float_data: 3.20000004768
  float_data: 1.29999995232
  float_data: 0.20000000298
}
protos {
  data_type: INT32
  int32_data: 0
}

This is the compact string that gets written into the db:

�̌@��L@ff�?��L>
"
```

```python
def write_db(db_type, db_name, features, labels):
    db = core.C.create_db(db_type, db_name, core.C.Mode.write)
    transaction = db.new_transaction()
    for i in range(features.shape[0]):
        feature_and_label = caffe2_pb2.TensorProtos()
        feature_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(features[i]),
            utils.NumpyArrayToCaffe2Tensor(labels[i])])
        transaction.put(
            'train_%03d'.format(i),
            feature_and_label.SerializeToString())
    # Close the transaction, and then close the db.
    del transaction
    del db

write_db("minidb", "iris_train.minidb", train_features, train_labels)
write_db("minidb", "iris_test.minidb", test_features, test_labels)
```

现在, 我们来创建一个简单的网络, 它包含一个 `TensorProtosDBInput` `operator`, 去说明如何从创造的`DB`中读取数据.
对于训练来说的话, 你可能创建的网络比较复杂: 创建一个网络, 训练它, 获取到模型, 然后执行预测. 为此, 你可以看一下 `MINIST tutorial`


```python

net_proto = core.Net("example_reader")
dbreader = net_proto.CreateDB([], "dbreader", db="iris_train.minidb", db_type="minidb")
net_proto.TensorProtosDBInput([dbreader], ["X", "Y"], batch_size=16)

print("The net looks like this:")
print(str(net_proto.Proto()))

```
```
The net looks like this:
name: "example_reader"
op {
  output: "dbreader"
  name: ""
  type: "CreateDB"
  arg {
    name: "db_type"
    s: "minidb"
  }
  arg {
    name: "db"
    s: "iris_train.minidb"
  }
}
op {
  input: "dbreader"
  output: "X"
  output: "Y"
  name: ""
  type: "TensorProtosDBInput"
  arg {
    name: "batch_size"
    i: 16
  }
}
```

```python
workspace.CreateNet(net_proto)

# Let's run it to get batches of features.
workspace.RunNet(net_proto.Proto().name)
print("The first batch of feature is:")
print(workspace.FetchBlob("X"))
print("The first batch of label is:")
print(workspace.FetchBlob("Y"))

# Let's run again.
workspace.RunNet(net_proto.Proto().name)
print("The second batch of feature is:")
print(workspace.FetchBlob("X"))
print("The second batch of label is:")
print(workspace.FetchBlob("Y"))
```

```
The first batch of feature is:
[[ 5.19999981  4.0999999   1.5         0.1       ]
 [ 5.0999999   3.79999995  1.5         0.30000001]
 [ 6.9000001   3.0999999   4.9000001   1.5       ]
 [ 7.69999981  2.79999995  6.69999981  2.        ]
 [ 6.5999999   2.9000001   4.5999999   1.29999995]
 [ 6.30000019  2.79999995  5.0999999   1.5       ]
 [ 7.30000019  2.9000001   6.30000019  1.79999995]
 [ 5.5999999   2.9000001   3.5999999   1.29999995]
 [ 6.5         3.          5.19999981  2.        ]
 [ 5.          3.4000001   1.5         0.2       ]
 [ 6.9000001   3.0999999   5.4000001   2.0999999 ]
 [ 6.          3.4000001   4.5         1.60000002]
 [ 5.4000001   3.4000001   1.70000005  0.2       ]
 [ 6.30000019  2.70000005  4.9000001   1.79999995]
 [ 5.19999981  2.70000005  3.9000001   1.39999998]
 [ 6.19999981  2.9000001   4.30000019  1.29999995]]
The first batch of label is:
[0 0 1 2 1 2 2 1 2 0 2 1 0 2 1 1]
The second batch of feature is:
[[ 5.69999981  2.79999995  4.0999999   1.29999995]
 [ 5.0999999   2.5         3.          1.10000002]
 [ 4.4000001   2.9000001   1.39999998  0.2       ]
 [ 7.          3.20000005  4.69999981  1.39999998]
 [ 5.69999981  2.9000001   4.19999981  1.29999995]
 [ 5.          3.5999999   1.39999998  0.2       ]
 [ 5.19999981  3.5         1.5         0.2       ]
 [ 6.69999981  3.          5.19999981  2.29999995]
 [ 6.19999981  3.4000001   5.4000001   2.29999995]
 [ 6.4000001   2.70000005  5.30000019  1.89999998]
 [ 6.5         3.20000005  5.0999999   2.        ]
 [ 6.0999999   3.          4.9000001   1.79999995]
 [ 5.4000001   3.4000001   1.5         0.40000001]
 [ 4.9000001   3.0999999   1.5         0.1       ]
 [ 5.5         3.5         1.29999995  0.2       ]
 [ 6.69999981  3.          5.          1.70000005]]
The second batch of label is:
[1 1 0 1 1 0 0 2 2 2 2 2 0 0 0 1]
```

## 总结

**两个关键类:**

* `TensorProtos` : 在制作的时候会用
* `TensorProtosDBInput`: 在读取的时候会用
* 同时可以看到, 每 `run` 一次 `net`, 都会从 `DB` 中读取新的数据.


## 参考资料
[https://caffe2.ai/docs/tutorial-create-your-own-dataset.html](https://caffe2.ai/docs/tutorial-create-your-own-dataset.html)
