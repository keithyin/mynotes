# Introduction to Chainer

这是 `Chainer Tutorial` 的第一部分。在这个部分，将会涉及到以下几个方面： 

- Simple example of forward and backward computation
- Usage of `links` and their gradient computation
- Construction of chains (a.k.a. “model” in most frameworks)
- Parameter optimization
- Serialization of links and optimizers

After reading this section, you will be able to:

- Compute gradients of some arithmetics
- Write a multi-layer perceptron with Chainer

## Core Concept

`Chainer` 对于神经网络来说是一个非常灵活的框架。因为它的主要目标是灵活性，所以它应该使得复杂的结构编写起来非常简单和直观。 

Most existing deep learning frameworks are based on the **“Define-and-Run”** scheme. That is, first a network is defined and fixed, and then the user periodically feeds it with mini-batches. Since the network is statically defined before any forward/backward computation, all the logic must be embedded into the network architecture as *data*. Consequently, defining a network architecture in such systems (e.g. Caffe) follows a declarative approach. Note that one can still produce such a static network definition using imperative languages (e.g. torch.nn, Theano-based frameworks, and TensorFlow).

### Define-by-Run

`Chainer` 采用 `Define-by-Run` 模式，网络在执行前向计算的时候建成。`Chainer`将网络表示成一个 `an execution path on a computation graph`。 计算图就是一系列的 `function applications`， 所以它可以由多个 `Function` 对象构成。当 `function` 是神经网络的层的时候，`function` 的 `parameters` 在训练的时候将会被更新。所以，`function` 需要将可训练的 `parameter` 保存起来，这就构成了 `Link` 类，`Chainer` 里的 `Link` 类就是将 `function` 和 `parameter` 打包保存了起来。



在之后的代码中，我们假设下面的 包 都已经被引入：

```python
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
```

## Forward/Backward Computation

`forward computation` 定义了神经网络。为了进行 前向计算，首先我们应该将 `input array` 设置成 `Variable` 对象。

```python
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
```

 [`Variable`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.Variable.html#chainer.Variable) 对象拥有基本的数学计算。下面代码用来计算$y=x^2−2x+1$

```
y = x**2 - 2 * x + 1
```

返回值 $y$ 依旧是 `Variable` 对象。可以通过 `.data` 来获取其值：

```python
>>> y.data
array([16.], dtype=float32)
```

`y` 不仅保存着结果值，同时也保存着计算的轨迹（即：计算图）。可以通过调用 `.backward()`来计算 参数的偏导。

```python
>>> y.backward()
```

反响传导的计算结果会保存在 `Variable` 的 `.grad` 属性中。

```python
>>> x.grad
array([8.], dtype=float32)
```

当 调用 `.backward` 的节点不是标量的时候，必须要指定其 梯度，指定的方法如下：

```python
>>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
>>> y = x**2 - 2*x + 1
>>> y.grad = np.ones((2, 3), dtype=np.float32)
>>> y.backward()
>>> x.grad
array([[ 0.,  2.,  4.],
       [ 6.,  8., 10.]], dtype=float32)
```

## Links

`Chainer` 对 神经网络中 层的概念的抽象 为`Link`，`Link` 将 `parameters` 和 `function` 整合了起来。 

最常用的 `Link` 是 `Linear Link`，也叫做全连接层或仿射变换。它代表的表达式为 $f(x)=xW^T+b$, 其中 $W,b$ 是参数。与这个 `Link` 对应的纯 function 为 `linear()` 。

```python
>>> f = L.Linear(3, 2)
```

`link` 的 `parameters` 作为 属性保存起来。每个 `parameter` 是 `Variable` 的一个实例。我们可以通过下列方法来访问 `link` 的 `parameter`：

```python
>>> f.W.data
array([[ 1.0184761 ,  0.23103087,  0.5650746 ],
       [ 1.2937803 ,  1.0782351 , -0.56423163]], dtype=float32)
>>> f.b.data
array([0., 0.], dtype=float32)
```

`Link` 的实例是个可调用对象，可以像函数一样调用：

```python
>>> x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
>>> y = f(x)
>>> y.data
array([[3.1757617, 1.7575557],
       [8.619507 , 7.1809077]], dtype=float32)
```

有些时候，计算输入数据 `shape` 是非常复杂的，`Chainer` 提供了一个机制，可以通过第一个 `mini-batch` 来计算出 数据的 `shape`。 

例如：我们可以通过下面的方式来创建一个`Link`实例，只指定了输出 特征个数，并没有指定输入特征个数，输入特征个数将从第一个 `mini-batch` 中推断出来

```python
f = L.Linear(2)
```

`.backward()` 会计算 `parameters` 的 梯度，注意，梯度是累加的，所以在调用 `.backward()` 之前，要记得将上一次计算的梯度清零。通过调用 `.cleargrad()` 来实现梯度清零：

```python
>>> f.cleargrads()
```

现在，我们可以通过调用 `.backward()` 来计算 梯度值：

```python
>>> y.grad = np.ones((2, 2), dtype=np.float32)
>>> y.backward()
>>> f.W.grad
array([[5., 7., 9.],
       [5., 7., 9.]], dtype=float32)
>>> f.b.grad
array([2., 2.], dtype=float32)
```

## Write a model as a chain

大多数的神经网络结构包含大量的 `links`。 例如，一个多层感知机由多个 线性层构成。

```python
>>> l1 = L.Linear(4, 3)
>>> l2 = L.Linear(3, 2)
>>> def my_forward(x):
...     h = l1(x)
...     return l2(h)
```

`L` 代表 `links` 模块。更加面向对象一些，可以将 `links` 和 前向过程写在一个类中

```python
class MyProc(object):
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)
    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)
```

为了更好的复用，我们希望支持 `parameter` 管理，`CPU/GPU`迁移，更加灵活和鲁棒的 `save/load` 特性。`Chainer` 通过 `Chain` 类来提供这些特性：

```python
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)
```

也可以通过继承 `ChainList` 来自定义 `MyChain` ：

```python
class MyChain2(ChainList):
    def __init__(self):
        super(MyChain2, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )
    def __call__(self, x):
        h = self[0](x)
        return self[1](h)
```

## Optimizer

已经介绍了如何搭建神经网络，为了优化神经网络参数，我们需要 `Optimizer` 类。给定 `Link`，`Optimizer` 来执行数值优化算法。很多数值优化算法都在 `optimizer` 模块有实现。

```python
model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)
```

`.setup()` 用来准备 `optimizer`。  

一些 `parameter/gradient` 操作，例如：`weight_decay, gradient_clipping` 通过给 `optimizer` 添加 `hook` 来实现。`hook functions` 在梯度计算后，更新执行前执行。

```python
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
```

当然，您也可以自定义 `hook functions`。 它应该是一个函数或者可调用对象，有一个 `optimizer` 参数。

`Chainer` 中包含两个使用 `optimizer` 的方法：

* 通过 `Trainer`
* 直接使用 

**直接使用也有两种方法：**

* 直接 `optimizer.update()`

```python
x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
# compute gradient here...
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
optimizer.update()
```

* 直接将 `loss` 函数传给 `optimizer`， 在这种方法中，`cleargrads()` 会被自动调用。

```python
def lossfun(arg1, arg2):
    # calculate loss
    loss = F.sum(model(arg1 - arg2))
    return loss
arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))
```



## Trainer

当我们想要训练神经网络时，我们需要 多次执行 *training loops* 更新参数，一个典型的 *training loop* 包含以下步骤：

1. 迭代训练数据集
2. 预处理数据然后取出一个 mini-batch
3. 执行 前向/后向 计算
4. 参数更新
5. 在验证集上评估当前的参数
6. 记录和打印中间结果

`Chainer` 对上面的 训练过程提供了非常简单的抽象，训练过程被抽象成两个单元：

- **Dataset abstraction**. It implements 1 and 2 in the above list. The core components are defined in the [`dataset`](https://docs.chainer.org/en/stable/reference/core/dataset.html#module-chainer.dataset) module. There are also many implementations of datasets and iterators in [`datasets`](https://docs.chainer.org/en/stable/reference/datasets.html#module-chainer.datasets) and [`iterators`](https://docs.chainer.org/en/stable/reference/iterators.html#module-chainer.iterators) modules, respectively.
- **Trainer**. It implements 3, 4, 5, and 6 in the above list. The whole procedure is implemented by [`Trainer`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer). The way to update parameters (3 and 4) is defined by [`Updater`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.training.Updater.html#chainer.training.Updater), which can be freely customized. 5 and 6 are implemented by instances of [`Extension`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.training.Extension.html#chainer.training.Extension), which appends an extra procedure to the training loop. Users can freely customize the training procedure by adding extensions. Users can also implement their own extensions.



## Serializer

`Serializer` 用来支持模型的保存和加载，`Link`，`Optimizer`， `Trainer` 都支持序列化。具体的 `serializer` 在 `serializers` 模块里面。支持， `Numpy， NPZ， HDF5` 格式。



```python
# 将 Link 对象序列化 NPZ 文件，它将 model 的参数保存在 my.model 文件中。
serializers.save_npz('my.model', model)

# 从序列化文件中 加载。
serializers.load_npz('my.model', model)
```

注意，只有 *parameters* 和 *persistent values* 会被 序列化。其它属性并不会自动序列化，如果想序列化其它数据，可以通过 `Link.add_persistent()` 将他们注册成 *persistent value* 。 



```python
# 同样的方法用来 load/save 优化器状态
serializers.save_npz('my.state', optimizer)
serializers.load_npz('my.state', optimizer)
```



## Example: Multi-layer Perceptron on MNIST

Now you can solve a multiclass classification task using a multi-layer perceptron (MLP). We use a hand-written digits dataset called [MNIST](http://yann.lecun.com/exdb/mnist/), which is one of the long-standing de facto “hello world” examples used in machine learning. This MNIST example is also found in the [examples/mnist](https://github.com/chainer/chainer/tree/master/examples/mnist)directory of the official repository. We show how to use [`Trainer`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer) to construct and run the training loop in this section.

We first have to prepare the MNIST dataset. The MNIST dataset consists of 70,000 greyscale images of size 28x28 (i.e. 784 pixels) and corresponding digit labels. The dataset is divided into 60,000 training images and 10,000 test images by default. We can obtain the vectorized version (i.e., a set of 784 dimensional vectors) by [`datasets.get_mnist()`](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.get_mnist.html#chainer.datasets.get_mnist).

```
>>> train, test = datasets.get_mnist()
...

```

This code automatically downloads the MNIST dataset and saves the NumPy arrays to the `$(HOME)/.chainer` directory. The returned `train` and `test` can be seen as lists of image-label pairs (strictly speaking, they are instances of [`TupleDataset`](https://docs.chainer.org/en/stable/reference/generated/chainer.datasets.TupleDataset.html#chainer.datasets.TupleDataset)).

We also have to define how to iterate over these datasets. We want to shuffle the training dataset for every *epoch*, i.e. at the beginning of every sweep over the dataset. In this case, we can use [`iterators.SerialIterator`](https://docs.chainer.org/en/stable/reference/generated/chainer.iterators.SerialIterator.html#chainer.iterators.SerialIterator).

```
>>> train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)

```

On the other hand, we do not have to shuffle the test dataset. In this case, we can pass `shuffle=False` argument to disable the shuffling. It makes the iteration faster when the underlying dataset supports fast slicing.

```
>>> test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

```

We also pass `repeat=False`, which means we stop iteration when all examples are visited. This option is usually required for the test/validation datasets; without this option, the iteration enters an infinite loop.

Next, we define the architecture. We use a simple three-layer rectifier network with 100 units per layer as an example.

```
>>> class MLP(Chain):
...     def __init__(self, n_units, n_out):
...         super(MLP, self).__init__()
...         with self.init_scope():
...             # the size of the inputs to each layer will be inferred
...             self.l1 = L.Linear(None, n_units)  # n_in -> n_units
...             self.l2 = L.Linear(None, n_units)  # n_units -> n_units
...             self.l3 = L.Linear(None, n_out)    # n_units -> n_out
...
...     def __call__(self, x):
...         h1 = F.relu(self.l1(x))
...         h2 = F.relu(self.l2(h1))
...         y = self.l3(h2)
...         return y

```

This link uses [`relu()`](https://docs.chainer.org/en/stable/reference/generated/chainer.functions.relu.html#chainer.functions.relu) as an activation function. Note that the `'l3'` link is the final linear layer whose output corresponds to scores for the ten digits.

In order to compute loss values or evaluate the accuracy of the predictions, we define a classifier chain on top of the above MLP chain:

```
>>> class Classifier(Chain):
...     def __init__(self, predictor):
...         super(Classifier, self).__init__()
...         with self.init_scope():
...             self.predictor = predictor
...
...     def __call__(self, x, t):
...         y = self.predictor(x)
...         loss = F.softmax_cross_entropy(y, t)
...         accuracy = F.accuracy(y, t)
...         report({'loss': loss, 'accuracy': accuracy}, self)
...         return loss

```

This Classifier class computes accuracy and loss, and returns the loss value. The pair of arguments `x` and `t` corresponds to each example in the datasets (a tuple of an image and a label).[`softmax_cross_entropy()`](https://docs.chainer.org/en/stable/reference/generated/chainer.functions.softmax_cross_entropy.html#chainer.functions.softmax_cross_entropy) computes the loss value given prediction and ground truth labels.[`accuracy()`](https://docs.chainer.org/en/stable/reference/generated/chainer.functions.accuracy.html#chainer.functions.accuracy) computes the prediction accuracy. We can set an arbitrary predictor link to an instance of the classifier.

The [`report()`](https://docs.chainer.org/en/stable/reference/util/generated/chainer.report.html#chainer.report) function reports the loss and accuracy values to the trainer. For the detailed mechanism of collecting training statistics, see [Reporter](https://docs.chainer.org/en/stable/reference/util/reporter.html#reporter). You can also collect other types of observations like activation statistics in a similar ways.

Note that a class similar to the Classifier above is defined as [`chainer.links.Classifier`](https://docs.chainer.org/en/stable/reference/generated/chainer.links.Classifier.html#chainer.links.Classifier). So instead of using the above example, we will use this predefined Classifier chain.

```
>>> model = L.Classifier(MLP(100, 10))  # the input size, 784, is inferred
>>> optimizer = optimizers.SGD()
>>> optimizer.setup(model)

```

Now we can build a trainer object.

```
>>> updater = training.StandardUpdater(train_iter, optimizer)
>>> trainer = training.Trainer(updater, (20, 'epoch'), out='result')

```

The second argument `(20, 'epoch')` represents the duration of training. We can use either `epoch`or `iteration` as the unit. In this case, we train the multi-layer perceptron by iterating over the training set 20 times.

In order to invoke the training loop, we just call the [`run()`](https://docs.chainer.org/en/stable/reference/core/generated/chainer.training.Trainer.html#chainer.training.Trainer.run) method.

```
>>> trainer.run()

```

This method executes the whole training sequence.

The above code just optimizes the parameters. In most cases, we want to see how the training proceeds, where we can use extensions inserted before calling the `run` method.

```
>>> trainer.extend(extensions.Evaluator(test_iter, model))
>>> trainer.extend(extensions.LogReport())
>>> trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
>>> trainer.extend(extensions.ProgressBar())
>>> trainer.run()  

```

These extensions perform the following tasks:

- [`Evaluator`](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.Evaluator.html#chainer.training.extensions.Evaluator)

  Evaluates the current model on the test dataset at the end of every epoch. It automatically switches to the test mode (see [Configuring Chainer](https://docs.chainer.org/en/stable/reference/core/configuration.html#configuration) for details), and so we do not have to take any special function for functions that behave differently in training/test modes (e.g. [`dropout()`](https://docs.chainer.org/en/stable/reference/generated/chainer.functions.dropout.html#chainer.functions.dropout), [`BatchNormalization`](https://docs.chainer.org/en/stable/reference/generated/chainer.links.BatchNormalization.html#chainer.links.BatchNormalization)).

- [`LogReport`](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.LogReport.html#chainer.training.extensions.LogReport)

  Accumulates the reported values and emits them to the log file in the output directory.

- [`PrintReport`](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.PrintReport.html#chainer.training.extensions.PrintReport)

  Prints the selected items in the LogReport.

- [`ProgressBar`](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.ProgressBar.html#chainer.training.extensions.ProgressBar)

  Shows the progress bar.

There are many extensions implemented in the [`chainer.training.extensions`](https://docs.chainer.org/en/stable/reference/extensions.html#module-chainer.training.extensions) module. The most important one that is not included above is [`snapshot()`](https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.snapshot.html#chainer.training.extensions.snapshot), which saves the snapshot of the training procedure (i.e., the Trainer object) to a file in the output directory.

The [example code](https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py) in the examples/mnist directory additionally contains GPU support, though the essential part is the same as the code in this tutorial. We will review in later sections how to use GPU(s).