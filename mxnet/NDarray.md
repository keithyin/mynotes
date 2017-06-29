# mxnet 学习笔记（一）



## NDArray

**NDArray**

* mxnet中参与计算的核心数据结构
* 用来表示一个 多维，固定大小的 array
* MXNET's NDArray executes code lazily，它可以自动并行
*  NDAarray 的属性
  * ndarray.shape 
  * ndarray.dtype
  * ndarray.size     : shape 返回的值的乘积
  * ndarray.context : 此 array 存放在哪个设备上。(cpu()  或 gpu(1))



## 创建NDArray

* 使用 python list 创建

  ```python
  import mxnet as mx
  # 一维 NDArray ，使用 python list 
  a = mx.nd.array([1,2,3])
  # 二维 NDArray ，使用 nested python list 
  b = mx.nd.array([[1,2,3], [2,3,4]])
  {'a.shape':a.shape, 'b.shape':b.shape}
  ```

* 使用 numpy 创建

  ```python
  import numpy as np
  import math
  c = np.arange(15).reshape(3,5)
  # 从 numpy.ndarray 对象中创建
  a = mx.nd.array(c)
  {'a.shape':a.shape}
  ```

* 使用 工具方法创建

  ```python
  # create a 2-dimensional array full of zeros with shape (2,3)
  a = mx.nd.zeros((2,3))
  # create a same shape array full of ones
  b = mx.nd.ones((2,3))
  # create a same shape array with all elements set to 7
  c = mx.nd.full((2,3), 7)
  # create a same shape whose initial content is random and
  # depends on the state of the memory
  d = mx.nd.empty((2,3))

  ```



## 打印NDArray

```python
b = mx.nd.arange(18).reshape((3,2,3))
b.asnumpy() 
```



## NDArray的一些操作

> 在 mx.nd 里面，+，-，×，/ 都已被重载，和numpy 的操作差不多。



## GPU支持

* 使用 `mx.Context()` 上下文管理器来指定 环境内 `operator` 的运行设备
* 或者可以直接在 `operator` 上指定

```python
gpu_device=mx.gpu() # Change this to mx.cpu() in absence of GPUs.
def f():
    a = mx.nd.ones((100,100))
    b = mx.nd.ones((100,100))
    c = a + b
    print(c)
# in default mx.cpu() is used
f()
# change the default context to the first GPU
with mx.Context(gpu_device):
    f()
```

```pytho
a = mx.nd.ones((100, 100), gpu_device)
```

```python
# cpu 与 gpu之间复制数据
a = mx.nd.ones((100,100), mx.cpu())
b = mx.nd.ones((100,100), gpu_device)
c = mx.nd.ones((100,100), gpu_device)
a.copyto(c)  # copy from CPU to GPU
d = b + c
e = b.as_in_context(c.context) + c  # same to above
{'d':d, 'e':e}
```



## 保存与加载数据（序列化）

* pickle

  ```python
  import pickle as pkl
  a = mx.nd.ones((2, 3))
  # pack and then dump into disk
  data = pkl.dumps(a)
  pkl.dump(data, open('tmp.pickle', 'wb'))
  # load from disk and then unpack
  data = pkl.load(open('tmp.pickle', 'rb'))
  b = pkl.loads(data)
  b.asnumpy()
  ```

* 通过 saver 和 load 方法来保存，保存的是二进制数据

  ```python
  # 可以保存一个 NDArray 也可以保存 list of NDArray
  a = mx.nd.ones((2,3))
  b = mx.nd.ones((5,6))
  mx.nd.save("temp.ndarray", [a,b])
  c = mx.nd.load("temp.ndarray")
  ```



## 懒惰执行和自动并行

`mxnet` 使用 懒惰执行策略来达到很好的 性能。



**什么是懒惰执行：**

* 当 python 执行 a=b+1, 这句指令的时候，python 线程仅仅将这句指令 push 给 后端引擎，然后返回，继续执行下一句指令。

**好处是什么呢？**

* 当一个指令被 push 给后端引擎后，python 主线程可以继续执行计算
* 对于 后端引擎来说，它可以去探索进一步的优化。例如：自动并行。



> 后端引擎可以解析数据之间的依赖，正确的调度计算。我们可以显式的 在结果 `array` 上调用方法 `wait_to_read()` 来等待计算完成。像 asnumpy()  就隐式调用了  `wait_to_read()`。



```python
import time
def do(x, n):
    """push computation into the backend engine"""
    return [mx.nd.dot(x,x) for i in range(n)]
def wait(x):
    """wait until all results are available"""
    for y in x:
        y.wait_to_read()  # 等待计算完 y 的值

tic = time.time()
a = mx.nd.ones((1000,1000))
b = do(a, 50)
print('time for all computations are pushed into the backend engine:\n %f sec' % (time.time() - tic))
wait(b)
print('time for all computations are finished:\n %f sec' % (time.time() - tic))
```



## 疑问？

* `mxnet` 在什么时候开始执行计算。还是 push 到后端之后，做一下自动并行就开始计算？

