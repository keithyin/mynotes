# 不使用 module 训练 mxnet 网络



**Symbol:**



| [`Symbol.bind`](http://mxnet.io/api/python/symbol.html?highlight=symbol#mxnet.symbol.Symbol.bind) | Binds the current symbol to an executor and returns it. |
| ---------------------------------------- | ---------------------------------------- |
| [`Symbol.simple_bind`](http://mxnet.io/api/python/symbol.html?highlight=symbol#mxnet.symbol.Symbol.simple_bind) | Bind current symbol to get an executor, allocate all the arguments needed. |

> bind 的作用是： Symbol 编程就像是定义了一个函数，然后我们编译这个函数。就有了一个实际的函数可以调用了。compile 的时候需要指定这个函数输入与输出，输入就是 bind 时候的参数，输出就是 执行 bind的 Symbol

**Symbol.bind**

`Symbol.bind(ctx, args, args_grad=None, grad_req='write', aux_states=None, group2ctx=None, shared_exec=None)`

* ctx： 上下文， 是啥设备
* args： NDArray 列表或 字典 'str':NDArray, 如果是 NDArray列表，对应的Symbol就是 list_arguments 顺序
* args_grad: NDArray 列表或 字典 'str':NDArray，如果提供的话，就用这个NDArray 来存储backward过程的梯度。
* grad_req: {'write', 'add', 'null'} 三选一，或 string list， 或者 dict  'str':'str' ,对 args_grad 的操作是重写，累加，还是不操作。
* 。。。
* 返回一个 Executor




**Symbol.simple_bind**

``simple_bind(ctx, grad_req='write', type_dict=None, group2ctx=None, shared_arg_names=None, shared_exec=None, shared_buffer=None, **kwargs)`

> Bind current symbol to get an executor, allocate all the arguments needed. Allows specifying data types.
>
> 此函数简化了 bind 的过程，仅仅需要指定 输入数据的形状

```python
x = mx.sym.Variable('x')
y = mx.sym.FullyConnected(x, num_hidden=4)
exe = y.simple_bind(mx.cpu(), x=(5,4), grad_req='null')
```






**Executor:**

* 用来执行前向过程，`y=executor.forward()` 得到的是 a list of NDArray。
* `e.outputs` 得到的也是  a list of NDArray，使用这个命令时，首先要 `forward` 一下。
* `executor` 的 `grad_arrays` 属性中保存着 Symbol 的梯度。
* `arg_arrays` 保存着参数列表 `NDArray`

```python
import mxnet as mx

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
d = 2 * a + b

print(d.list_arguments())
type(d)

e = d.simple_bind(mx.cpu(), a=(1,), b=(1,)) # e是一个 Executor

e.copy_params_from(arg_params = {'a':mx.nd.array([1.]), 'b':mx.nd.array([2.])})

y = e.forward(is_train=True)

e.backward(out_grads=mx.nd.array([1.]))
```



## 几个参数

* `param_names`: `arg_names` 去掉 `input_names`
* `arg_names`: 计算定义过程中，所有定义的 `Variable`。
* `input_names`: `data_names` +`label_names`+`state_names`
* `data_names`: 
* `label_names`:
* `state_names`:
* `param_arrays`: Executor 中的 `arg_arrays` 已经对所有的 ` arg_names` 都创建了一个 `NDArray`。 `param_arrays` 保存 具有 `param_names` 的这部分。 

## 思考

`mxnet` 在定义完 计算图，然后 `bind` 之后，所有的 `Symbol` 都会变成 `NDArray`，然后就在`Executor` 里面用这个 `NDArray`计算了。

```python
import mxnet as mx

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
d = 2 * a + b
e = d.simple_bind(mx.cpu(), a=(1,), b=(1,))

y = e.forward()

print(e.outputs[0].asnumpy())
# e.backward(out_grads=mx.nd.array([2.]))
for value in e.arg_arrays:
    print(value.asnumpy())
    value += mx.nd.array([1])
e.forward()

for value in e.outputs:
    print(value.asnumpy())
```



## 参考资料

[http://mxnet.io/api/python/symbol.html?highlight=symbol#module-mxnet.symbol](http://mxnet.io/api/python/symbol.html?highlight=symbol#module-mxnet.symbol)

