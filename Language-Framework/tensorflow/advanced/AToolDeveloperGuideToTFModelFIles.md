# Tensorflow Model Files

最近闲来无聊，想深入理解一下`tensorlfow`，也不知从何下手，突然间发现了官方文档的`Extend`模块下还有这个一片文章 `A Tool Developer's Guide to TensorFlow Model Files`, 所以就打算边翻译，边学习了。水平有限，如发现错误，请不吝指出！



## 翻译开始

大多数用户不需要关心`tensorflow`在硬盘上存储数据的细节问题的，但是如果你是一个 `Tool developer`， 那就另当别论了。例如，如果你想分析模型(`models`)，或者想在`tensorflow`或者其它格式之间进行来回转换。这篇指南通过试着去解释一些 如何处理 `保存着模型数据的文件的`细节，使得开发者们做一些格式装换的工具更加简单。



## Protocol Buffers

所有的`Tensorflow`的文件格式都是基于[Protocol Buffers](https://developers.google.com/protocol-buffers/?hl=en)的。所以了解它们是如何工作的是非常有价值的。概括来说就是，你在文本文件(`text files`)中定义数据结构，`protobuf tools`就会生成对应的`C,Python和其它语言`的类。我们可以用友好的方式来加载，保存，访问这些类中的数据。我们经常将 `Protocol Buffers`称为 `protobufs`，在接下来的文章中，我们将继续遵守这个约定。



## GraphDef

在`tensorflow`中，计算的基础是`Graph`对象。`Graph`对象保存着网络的节点，每个节点代表一个`Operation`(`add, matmul, etc`)，节点之间由输入和输出链接起来。当建好了一个`Graph`对象之后，可以通过`Graph.as_graph_def()` 把它保存起来，`as_graph_def()` 返回一个 `GraphDef`对象。



`GraphDef`类 是由`ProtoBuf`库创建的对象。它的定义在[tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)。 `protobuf tools`解析这个文本文件，然后生成代码用来加载，存储，和操作图定义。如果看到一个独立的  用于表示模型(`model`)的`Tensorflow`文件，那么它很可能是 由`protobuf code` 保存的序列化的`GraphDef`对象。



`protobuf code` 用来从硬盘上 保存和加载`GraphDef`对象。加载对象的代码看起来像是这样:

```python
#这行代码创建了一个空的 GraphDef 对象。GraphDef类已经由 graph.proto 中定义的文本 所创建。
#我们将用文本中的数据来填充这个对象
graph_def = tf.GraphDef()

if FLAGS.input_binary:
    with open("graph_def.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
else:
    with open("graph_def.pb", mode='r') as f
        text_format.Merge(f.read(), graph_def)
```

> 译者注：`txt_format`是一个工具模块，在`tensorflow`中，但是木有找到。
>
> 这里只是掩饰了如何load `ProtoBuf`，但是，并没有说明如何保存`ProtoBuf`，如果想要保存的话，`tensorflow`提供了一个接口 `tf.train.write_graph(graph_def, "./", name='graph.pb')`。用这个就可以保存成`ProtoBuf`。
>
> 当然，加载的话，`tensorflow`也提供了一个接口:
>
> ```python
> def import_graph_def(graph_def, input_map=None, return_elements=None,
>                      name=None, op_dict=None, producer_op_list=None)
>
> ```

## Text or Binary

有两种不同的文件格式可以存储`ProtoBuf`。一个是`TextFormat`，人类可以很容易的理解，而且可以很容易的进行`debugging`或者`editing`，但是如果里面包含数值数据的话，那么这个文件就会变的很大。这里有一个例子 [graph_run_run2.pbtxt](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/components/tf_tensorboard/test/data/graph_run_run2.pbtxt) 尴尬的是，官方给的这个例子找不到了。。。

另一种文件格式是 `BinaryFormat`，它比`TextFormat`所需的存储空间小，但是人类读不懂。在上面提供的脚本文件中，我们要求用户提供 `flag` 用来指示，我们读取的文件是 `TextFormat`还是`BinaryFormat`，这样我们才能够找到正确的方法去调用。这里有一个`BinaryFormat`的例子[inception_v3 archive](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz) `inception_v3_2016_08_28_frozen.pb`.

不过`API`的设计着实让人懵逼-对于`BinaryFormat` ，我们调用 `ParseFromString()`, 对于`TextFormat`，我们使用`text_format`模块。



## Nodes

一旦将文件加载到`graph_def`对象，你就可以访问内部的数据了。出于实用目的，最重要的部分是存储`节点`成员的节点列表。下面的循环代码可以获取到它们：

```python
for node in graph_def.node:
    print(node)
```

每个节点(`node`)是一个`NodeDef`对象，定义在[tensorflow/core/framework/node_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto).这些节点是`Tensorflow`中`Graph`的基本构件块，每个都定义了一个`operation`和它的输入连接。

下面将介绍 `NodeDef`的成员和其所代表的含义。

**name**

每个节点(`Node`) 应该有一个唯一的标识符，图中的其它节点不能使用该标识符(这个标识符就是`name`属性对应的值)。在使用`tensorflow Python`接口的时候，如果没有显示指定`name`属性，那么`tensorflow`会自动选择一个`name`，`name`的格式是 `operation_name`加上一个累加的数字。

`name`用来定义节点之间的连接 ，和在运行时为整个图形设置输入输出。



**op**

这个属性指明要执行哪个`operation`，例如`"Add"`, `"MatMul"`, 或者 `"Conv2D"`。当`Graph`运行起来的时候，就会在注册表中查找这些`op`的名称以找到其对应的实现。注册表是通过调用`REGISTER_OP()` 宏来填充的，就像这些[tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc).



**input**

一个`strings`列表，列表中的每个元素是其它节点的名字，可选的在后面跟上一个冒号和输出端口号。例如：一个拥有两个输入的节点的`input`属性大概是这样的`["some_node_name", "another_node_name"]`, 等价于`["some_node_name:0", "another_node_name:0"]`,说明了，当前`node`的第一个输入是名字为`"some_node_name"`的`Node`的第一个输出，当前`node`的第二个输入是名字为`"another_node_name"`的`Node`的第一个输出。

> 我的测试结果是，现在的input在pdtxt中是下面这种形式，而不是文档中所说的 `strings list`
>
> input: "some_node_name"
>
> input: "another_node_name"



**device**

多数情况下，可以忽略这东西。它规定了在分布式情况下，哪个设备执行这个节点，或者是你想强制一个`operation`在`CPU`上或是`GPU`上运行。



**attr**

这个属性保存了`key/value`键值对，用来指定节点的所有属性。这是一个节点的 永久属性，一旦指定，在运行时刻就不能再被修改了，例如：卷积核的大小，或者是`constant op` 的值。 由于可能有多种不同类型的属性值，从`strings`，到`int`，再到`tensor 值的 arrays`。这里有单独的`protobuf file`文件，定义着这些数据结构[tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto).



每个属性拥有一个唯一的名字字符串，在定义`operation`的时候，期望的属性会被列出来。当一个属性没有在`node`中出现时，但是在定义`op`的时候，它有一个属性的默认值，那么这个默认值将会在创建图的时候使用。



在`Python`中，你可以 通过调用 `node.name, node.op, etc` 访问所有的这些成员 。在`GraphDef`中存储的 节点列表是模型体系结构的完整定义。



## Freezing

令人困惑的一点是 **在训练过程中，权值通常不保存在 file format 中**。 相反，它们被保存在单独地 检查点`checkpoint`文件中，初始化时，图中的`Variable op`用于加载最近的值。在部署到生产环境的时候，用于单独的文件通常会不方便。所以，这里有一个[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)脚本文件，用于将 `graph definition`和 一组`checkpoints` 冻结成一个文件。



它是怎么做的呢？加载`GraphDef`，将所有的变量从最近的 检查点文件中取出，然后将`GraphDef`中的`Variable op` 替换成 `Const op`, 这些`Const op`中保存着 检查点中保存的变量的值。然后，它去掉`GraphDef`中与 前向过程无关的节点，然后将处理后的`GraphDef`保存到输出文件中。



> 部署的时候，用这个玩意感觉爽的很。



## Weight Formats

如果你正在处理一些 表示神经网络的 `TensorFlow`模型，最常见的问题之一就是 提取和 解释权重值。存储它们的常用方法就是，用`freeze_graph`脚本处理`GraphDef`，将`Variable op` 换成 `Const op`，使用`Const op`将这些权重作为`Tensor`存储起来。`Tensor`被定义在[tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto), `Tensor` 中不仅保存了权重的值，还保存了数据类型(`int,float`)和`size`。在`Python`中，可以通过表示 `Const op`的 `NodeDef`对象中获取`TensorProto`对象，就像  

```python
tensorProto = some_node_def.attr['value'].tensor
```



这段代码会返回一个 表示权重数据的对象。数据本身会保存在一个列表中，这个列表的名字是`suffix_val`, `suffix`代表对象的数据类型，例如`float_val` 代表 32位浮点型。



当在不同的框架之间进行转换时，卷积权重的顺序是很难处理的。在`Tensorflow`中，`Conv2D op`的卷积核的存储在第二个输入上，期望的顺序是`[filter_height, filter_width, input_depth, output_depth]`，在这里，`filter_count`增加一意味着移动到内存中的相邻值。



希望这个纲要能让你更好地了解`TensorFlow`模型文件中正在发生的事情，如果你需要对它们进行操作的话，将会对你有所帮助。



## 翻译完毕，总结

本文中提到了以下几个概念：

* `protobuf`
  * 文中提到了`protobuf code`, `protobuf` 
  * `protobuf code` 指的应该是解析`protobuf`文件的代码
  * `protobuf`指的应该是，官方写的`proto`文件，其中描述了一些类的定义
  * 剩下的就是我们自己使用` protobuf code` 处理的模型定义了。
* `GraphDef`
  * `GraphDef`中存储的节点列表是模型体系结构的完整定义
* `NodeDef`
  * 用于代表一个`op`及其 输入输出
  * `name`: `name`属性表示`op`的名字 `name:ouput_index`代表输出`tensor`
  * `input`属性用于暴露`op`的输入



## Demo

下面只是给出了一个简单的代码，[这里也有一个示例](https://github.com/tensorflow/tensorflow/issues/616)。



**保存为pb**

```python
import tensorflow as tf
t = tf.constant([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
paddings = tf.constant([[1,0], [2,2], [1,2]])

paded = tf.pad(t, paddings, "CONSTANT")

graph_def = tf.get_default_graph().as_graph_def()
print(graph_def)

tf.train.write_graph(graph_def, logdir="./", name='graph.pb', as_text=True)
```

**打印出来的结果为：**

```
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\006\000\000\000\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\006\000\000\000"
      }
    }
  }
}
node {
  name: "Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 3
          }
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\000\000\000\000\002\000\000\000\002\000\000\000\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "Pad"
  op: "Pad"
  input: "Const"
  input: "Const_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tpaddings"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 21
}
```



**解析pb**

```python
import tensorflow
from google.protobuf import text_format

graph_def = tf.GraphDef()
#因为是文本文件，所以mode='r'，如果之前保存的是二进制文件 mode='rb'
with open("./graph.pb", mode='r') as file:
    text_format.Merge(file.read(), graph_def)

tf.import_graph_def(graph_def=graph_def, name='')

#get_tensor_by_name有一个需要注意的地方，就是 tensor的name需要是 op_name:output_index
padded = tf.get_default_graph().get_tensor_by_name("Pad:0")

with tf.Session() as sess:
    print(sess.run(padded))
```



**当我们用这种方式只进行推断的时候，我们可以这么做：**

* 获取`placeholder`   `tensor`
* `feed` 这些 `tensor`值
* 获取最后一层的`tensor`，然后`sess.run`打印出来结果就 `OK`



## 最后说明一下前面用到的几个方法

```python
def import_graph_def(graph_def, input_map=None, return_elements=None,
                     name=None, op_dict=None, producer_op_list=None)
# name : 可选的，加在GraphDef中名字的前面，默认是import ，一般情况下，直接 name=''就可以了
# input_map: 没有测试到底是干嘛的，默认值就可以。

tf.train.write_graph(graph_or_graph_def, logdir, name, as_text=True)
# logdir: 导出的文件目录
# name: 导出时的文件名
# as_text: 是以Text形式 还是 binary 形式导出， 默认为True
```

