# load and save

**当序列化 NDArray 的时候，我们序列化的是NDArray 中保存的 tensor 值。当序列化 Symbol 的时候，我们序列化的是 Graph。**

## Symbol序列化

当序列化 `Symbol` 的时候，通常使用 `json` 文件作为序列化后的文件，因为可读性好。

```python
import mxnet as mx
a = mx.sym.Variable('a', shape=[2,])
b = mx.sym.Variable('b', shape=[3,])
c = a+b
print(c.tojson()) # 打印出来 json 文件，看看里面是啥
c.save('symbol-c.json') # 保存文件

c2 = mx.sym.loads('symbol-c.json') # 加载 json 文件，此时 c2 就代表一个 symbol
```

```json

{
  "nodes": [
    {
      "op": "null", 
      "name": "a", 
      "attr": {"__shape__": "[2]"}, 
      "inputs": []
    }, 
    {
      "op": "null", 
      "name": "b", 
      "attr": {"__shape__": "[3]"}, 
      "inputs": []
    }, 
    {
      "op": "elemwise_add", 
      "name": "_plus0", 
      "inputs": [[0, 0, 0], [1, 0, 0]]
    }
  ], 
  "arg_nodes": [0, 1], 
  "node_row_ptr": [0, 1, 2, 3], 
  "heads": [[2, 0, 0]], 
  "attrs": {"mxnet_version": ["int", 1000]}
}
```

* heads ： 表示输出
* [2, 0, 0], [1, 0, 0] 这些应该是表示的 Symbol 的 id。



## NDArray 序列化

ndarray 序列化是序列化 ndarray 中的 tensor 值。

序列化 NDArray 有两种方法：

* 使用 pickle ， （python）
  * 序列化：pkl.dumps()  pkl.dump() 
  * 加载：pkl.load(), pkl.loads()
* 使用 NDArray 自带的 方法
  * 序列化：mx.nd.save() 
  * 加载：mx.nd.load()

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

a = mx.nd.ones((2,3))
b = mx.nd.ones((5,6))
mx.nd.save("temp.ndarray", [a,b])
c = mx.nd.load("temp.ndarray")
c

d = {'a':a, 'b':b}
mx.nd.save("temp.ndarray", d)
c = mx.nd.load("temp.ndarray")
c
```



## Module 保存参数与加载参数

### 保存

使用 checkpoint callback 在每个 epoch 之后保存一次参数。

```python
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)
```

**如果不用 fit 的话，如何保存呢？**

先看下fit部分的代码

```python
# sync aux params across devices
arg_params, aux_params = self.get_params()
self.set_params(arg_params, aux_params)

if epoch_end_callback is not None:
    for callback in _as_list(epoch_end_callback):
        callback(epoch, self.symbol, arg_params, aux_params)
```

我们只需要模拟这部分代码，手动调用 `callback` 就可以了

```python
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)

# ...
mod.bind(...)

# 调用这个函数来 保存参数就可以了
def save_checkpoint(epoch, module, callback):
    arg_params, aux_params = module.get_params()
    module.set_params(arg_params, aux_params)
    callback(epoch, module.symbol, arg_params, aux_params)
```






### 加载

加载保存了的 模型参数，使用 `load_checkpoint` 方法

```python
# 不仅加载了 参数，同时加载了 Symbol
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
assert sym.tojson() == net.tojson()

# 然后创建一个 module
# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)
```



## 参考资料

[https://mxnet.incubator.apache.org/tutorials/basic/module.html#save-and-load](https://mxnet.incubator.apache.org/tutorials/basic/module.html#save-and-load)

[https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html#serialize-from-to-distributed-filesystems](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html#serialize-from-to-distributed-filesystems)

[https://mxnet.incubator.apache.org/tutorials/basic/symbol.html#load-and-save](https://mxnet.incubator.apache.org/tutorials/basic/symbol.html#load-and-save)

