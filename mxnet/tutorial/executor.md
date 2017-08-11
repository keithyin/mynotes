# mxnet Executor

当我们执行 `Symbol.bind ` 时候，会返回 一个 `Executor` 对象。

`Executor` 的 forward 和 backward 是按照自己的内部状态来运行的。

## 属性

* `Executor.arg_arrays`
* `Executor.grad_arrays`
* `Executor.aux_arrays`


* `Executor.arg_dict` 返回 dict ，参数名: 值
* `Executor.grad_dict` 
* `Executor.aux_dict`
* `Excutor.output_dict`



## 方法

* `Executor.forward`
* `Executor.backward`
* `Executor.copy_params_from`
* `Executor.reshape`
* `Executor.debug_str`

