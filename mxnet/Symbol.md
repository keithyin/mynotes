# mxnet 学习笔记（二）：Symbol

在 `mxnet` 中，仅仅使用 `NDArray` 是可以 以 命令式编程的风格编写出一个 用于科学计算的程序。

但 `mxnet` 同时提供了 `symbolic programming` 的接口。在命令式编程中，程序的执行是 `step by step` 的，但是在 `symbolic programming` 中，我们是首先定义好一个计算图。计算图中包含：

* placeholders : 用来接收输入
* 指明的输出



`mxnet` 符号编程的一般套路为：

1. 定义计算图
2. 编译计算图，产出一个函数
3. 将这个函数与 `NDArray` 绑定，然后执行。



## 定义计算图

* 使用 `mx.sym.Variable()` 创建 `placeholder`（即：占坑符）

  ```python
  import mxnet as mx
  a = mx.sym.Variable('a')  #占坑符，名字为a
  b = mx.sym.Variable('b')
  c = a + b #不指定 Symbol名字的话，mxnet 会自动指定一个唯一的名字
  # elemental wise multiplication
  d = a * b
  # matrix multiplication
  e = mx.sym.dot(a, b)
  # reshape
  f = mx.sym.reshape(d+e, shape=(1,4))
  # broadcast
  g = mx.sym.broadcast_to(f, shape=(2,4))
  # plot
  mx.viz.plot_network(symbol=g).render() #这个就可以看到计算图被打印出来咯
  ```

* 支持 `NDArray` 的 `operator` 同样也支持 `Symbol`

* `mx.sym` 中同时也包含了大量的**神经网络层**：

  ```python
  # 每个 Symbol 都有一个唯一的 string name。Symbol和NDArray都表示一个Tensor
  # Operators 代表 Tensor 之间的计算。
  net = mx.sym.Variable('data')
  net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
  net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
  net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
  net = mx.sym.SoftmaxOutput(data=net, name='out')
  mx.viz.plot_network(net, shape={'data':(100,200)}) #用来画计算图

  #可以将 Symbol 简单的看作为一个 接收多个参数的函数，下面方法打印出来，Symbol 的参数
  net.list_arguments() 
  ```

* **注意 mx.sym 与 mx.symbol 是指的同一个包**

* 也可以手动定义 weight或  bias

  ```python
  net = mx.symbol.Variable('data')
  #网络参数也用Variable指定，追下源码可以看到，
  # Variable 实际上也是返回一个 Symbol
  w = mx.symbol.Variable('myweight') 

  #手动指定的话，就是手动指定，没有指定的就会用默认的，
  net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)
  net.list_arguments()
  ```



**碰到的第二个上下文管理器：**

* mx.name.Prefix(str): 用来给Symbol名字加前缀的。





> mxnet 中，并没有 模型参数和 网络中流动的数据做区分呢。



**将 Symbol 打包**

```python
net = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
out1 = mx.sym.SoftmaxOutput(data=net, name='softmax')
out2 = mx.sym.LinearRegressionOutput(data=net, name='regression')
group = mx.sym.Group([out1, out2])
group.list_outputs()
```



## Symbol 与 NDArray

**NDArray:**

* 提供了命令式编程接口
* 很直观
* 与本地语言可以有很好的交互，(for...loop, if-else condition, ...)
* 容易 step-by-step 调试

**Symbol:**

* 和 NDArray 拥有共同的计算函数
* 容易保存，加载，可视化。
* 容易后端去优化
* 先声明一个计算图，然后将计算图和数据绑定起来 运行。



## Symbol 操作

**形状和类型推断：**

```python
arg_name = c.list_arguments()  # get the names of the inputs
out_name = c.list_outputs()    # get the names of the outputs
# infers output shape given the shape of input arguments
arg_shape, out_shape, _ = c.infer_shape(a=(2,3), b=(2,3))
# infers output type given the type of input arguments
arg_type, out_type, _ = c.infer_type(a='float32', b='float32')
{'input' : dict(zip(arg_name, arg_shape)),
 'output' : dict(zip(out_name, out_shape))}
{'input' : dict(zip(arg_name, arg_type)),
 'output' : dict(zip(out_name, out_type))}
```



**与数据绑定：**

```python
device = mx.gpu()
# device = mx.cpu()
ex = c.bind(ctx=device, args={'a' : mx.nd.ones([2,3]),
                                'b' : mx.nd.ones([2,3])})
ex.forward()
print('number of outputs = %d\nthe first output = \n%s' % (
           len(ex.outputs), ex.outputs[0].asnumpy()))
```



## 保存与加载

当序列化 `NDArray` 的时候，会把 `NDArray` 中的值保存。

保存`Symbol` 的时候，也可以用 `save` 和 `load`

```python
print(c.tojson())
c.save('symbol-c.json')
c2 = mx.sym.load('symbol-c.json')
c.tojson() == c2.tojson()
```



## 高级用法

**数据类型转换：**

```python
a = mx.sym.Variable('data')
b = mx.sym.cast(data=a, dtype='float16')
arg, out, _ = b.infer_type(data='float32')
print({'input':arg, 'output':out})

c = mx.sym.cast(data=a, dtype='uint8')
arg, out, _ = c.infer_type(data='int32')
print({'input':arg, 'output':out})
```



**变量共享：**

```python
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
b = a + a * a

data = mx.nd.ones((2,3))*2
ex = b.bind(ctx=mx.cpu(), args={'a':data, 'b':data}) # 共享相同的内容。
ex.forward()
ex.outputs[0].asnumpy()
```



## 参考资料

[http://mxnet.io/tutorials/basic/symbol.html](http://mxnet.io/tutorials/basic/symbol.html)

