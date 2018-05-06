# learn mxnet : 1

* Symbolic on network definition


* Imperative on tensor computation

从这两句可以看出， `mxnet` 的符号编程只是用来定义网络的，定义好了网络，计算的时候依旧是 命令式执行的。

* `Symbol.list_arguments()` 返回所有 `Variable` 的名字
* `Symbol.list_outputs()` 返回 输出的名字。
* 更可以看出来，`Symbol` 是一个函数



## 从 bind 来看 mxnet

* bind 将定义的网络 编译成一个函数，函数中的所有参数（包含网络输入和网络参数）都是这个函数的 输入。

**先来看执行Symbol.simple_bind**

```python
    def simple_bind(self, ctx,
                    grad_req='write',
                    type_dict=None,
                    group2ctx=None,
                    **kwargs):
        """Binds current symbol to get an executor, allocate all the arguments needed.

        This function simplifies the binding procedure. You need to specify only input data shapes.
        Before binding the executor, the function allocates arguments and auxiliary states
        that were not explicitly specified. Allows specifying data types.

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        grad_req: string
            {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            To specify how we should update the gradient to the `args_grad`.

            - 'write' means every time gradient is written to specified `args_grad` NDArray.
            - 'add' means every time gradient is added to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        type_dict  : Dict of str->numpy.dtype
            Input type dictionary, name->dtype

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        kwargs : Dict of str->shape
            Input shape dictionary, name->shape

        Returns
        -------
        executor : mxnet.Executor
            The generated executor
        """
        # pylint: disable=too-many-locals
        if type_dict is None:
            attrs = self.attr_dict()
            type_dict = {k: mx_real_t for k in self.list_arguments()
                         if k not in attrs or '__dtype__' not in attrs[k]}
        # 所有 Variable 的形状，辅助参数的形状。
        arg_shapes, _, aux_shapes = self.infer_shape(**kwargs)
        arg_types, _, aux_types = self.infer_type(**type_dict)

        if arg_shapes is None or arg_types is None:
            raise ValueError("Input node is not complete")

        if group2ctx is not None:
            attr_dict = self.attr_dict()
            arg_ctx = [group2ctx.get(attr_dict[name]['__ctx_group__'], ctx) \
                         if name in attr_dict and '__ctx_group__' in attr_dict[name] \
                         else ctx for name in self.list_arguments()]
            aux_ctx = [group2ctx.get(attr_dict[name]['__ctx_group__'], ctx) \
                         if name in attr_dict and '__ctx_group__' in attr_dict[name] \
                         else ctx for name in self.list_auxiliary_states()]
        else:
            arg_ctx = [ctx] * len(arg_shapes)
            aux_ctx = [ctx] * len(aux_shapes)

        # alloc space,对所有的 Variable。不包含 辅助参数。
        arg_ndarrays = [
            _nd_zeros(shape, dev, dtype=dtype)
            for dtype, dev, shape in zip(arg_types, arg_ctx, arg_shapes)]
        # 初始化 存放梯度的空间。
        if grad_req != 'null':
            grad_ndarrays = {}
            for name, shape, dev, dtype in zip(
                    self.list_arguments(), arg_shapes, arg_ctx, arg_types):
                if not isinstance(grad_req, dict) or grad_req[name] != 'null':
                    grad_ndarrays[name] = _nd_zeros(shape, dev, dtype=dtype)
        else:
            grad_ndarrays = None
		
        # 辅助参数 也 分配了空间。
        aux_ndarrays = [_nd_zeros(shape, dev, dtype=dtype)
                        for shape, dev, dtype in zip(aux_shapes, aux_ctx, aux_types)]
        executor = self.bind(ctx, arg_ndarrays,
                             grad_ndarrays, grad_req, aux_ndarrays,
                             group2ctx=group2ctx)
        return executor
```

* 给 所有的 `Variable` 和辅助 参数 分配了空间。`arg_ndarrays, aux_ndarrays`
* 给 `Variable` 分配了存放 梯度的空间。 `grad_ndarrays`
* 既然所有都初始化成 ndarray 了，可以调用 Symbol.bind() 了

**Symbol.bind**

```python
    def bind(self, ctx, args, args_grad=None, grad_req='write',
             aux_states=None, group2ctx=None, shared_exec=None):
        """Binds the current symbol to an executor and returns it.

        We first declare the computation and then bind to the data to run.
        This function returns an executor which provides method `forward()` method for evaluation
        and a `outputs()` method to get all the results.
        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        args : list of NDArray or dict of str to NDArray
            Input arguments to the symbol.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_arguments()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
              to the corresponding `NDArray`.
            - In either case, all the arguments must be provided.

        args_grad : list of NDArray or dict of str to `NDArray`, optional
            When specified, `args_grad` provides NDArrays to hold
            the result of gradient value in backward.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_arguments()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
              to the corresponding NDArray.
            - When the type is a dict of str to `NDArray`, one only need to provide the dict
              for required argument gradient.
              Only the specified argument gradient will be calculated.

        grad_req : {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            To specify how we should update the gradient to the `args_grad`.

            - 'write' means everytime gradient is write to specified `args_grad` `NDArray`.
            - 'add' means everytime gradient is add to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        aux_states : list of `NDArray`, or dict of str to `NDArray`, optional
            Input auxiliary states to the symbol, only needed when the output of
            `list_auxiliary_states()` is not empty.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_auxiliary_states()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of
              `auxiliary_states` to the corresponding `NDArray`,
            - In either case, all the auxiliary states need to be provided.

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        shared_exec : mx.executor.Executor
            Executor to share memory with. This is intended for runtime reshaping, variable length
            sequences, etc. The returned executor shares state with `shared_exec`, and should not be
            used in parallel with it.

        Returns
        -------
        executor : Executor
            The generated executor

        Notes
        -----
        Auxiliary states are the special states of symbols that do not correspond
        to an argument, and do not have gradient but are still useful
        for the specific operations. Common examples of auxiliary states include
        the `moving_mean` and `moving_variance` states in `BatchNorm`.
        Most operators do not have auxiliary states and in those cases,
        this parameter can be safely ignored.

        One can give up gradient by using a dict in `args_grad` and only specify
        gradient they interested in.
        """
        # pylint: disable=too-many-locals, too-many-branches
        if not isinstance(ctx, Context):
            raise TypeError("Context type error")

        listed_arguments = self.list_arguments()
        args_handle, args = self._get_ndarray_inputs('args', args, listed_arguments, False)
        # setup args gradient
        if args_grad is None:
            args_grad_handle = c_array(NDArrayHandle, [None] * len(args))
        else:
            args_grad_handle, args_grad = self._get_ndarray_inputs(
                'args_grad', args_grad, listed_arguments, True)

        if aux_states is None:
            aux_states = []
        aux_args_handle, aux_states = self._get_ndarray_inputs(
            'aux_states', aux_states, self.list_auxiliary_states(), False)

        # setup requirements
        if isinstance(grad_req, string_types):
            if grad_req not in _GRAD_REQ_MAP:
                raise ValueError('grad_req must be in %s' % str(_GRAD_REQ_MAP))
            reqs_array = c_array(
                mx_uint,
                [mx_uint(_GRAD_REQ_MAP[grad_req])] * len(listed_arguments))
        elif isinstance(grad_req, list):
            reqs_array = c_array(mx_uint, [mx_uint(_GRAD_REQ_MAP[item]) for item in grad_req])
        elif isinstance(grad_req, dict):
            req_array = []
            for name in listed_arguments:
                if name in grad_req:
                    req_array.append(mx_uint(_GRAD_REQ_MAP[grad_req[name]]))
                else:
                    req_array.append(mx_uint(0))
            reqs_array = c_array(mx_uint, req_array)

        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []

        if group2ctx:
            for key, val in group2ctx.items():
                ctx_map_keys.append(c_str(key))
                ctx_map_dev_types.append(ctypes.c_int(val.device_typeid))
                ctx_map_dev_ids.append(ctypes.c_int(val.device_id))

        handle = ExecutorHandle()
        shared_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()
        check_call(_LIB.MXExecutorBindEX(self.handle,
                                         ctypes.c_int(ctx.device_typeid),
                                         ctypes.c_int(ctx.device_id),
                                         mx_uint(len(ctx_map_keys)),
                                         c_array(ctypes.c_char_p, ctx_map_keys),
                                         c_array(ctypes.c_int, ctx_map_dev_types),
                                         c_array(ctypes.c_int, ctx_map_dev_ids),
                                         mx_uint(len(args)),
                                         args_handle,
                                         args_grad_handle,
                                         reqs_array,
                                         mx_uint(len(aux_states)),
                                         aux_args_handle,
                                         shared_handle,
                                         ctypes.byref(handle)))
        executor = Executor(handle, self, ctx, grad_req, group2ctx)
        # args 和 excutor.arg_arrays 指向了一个对象
        # args_grad 和 executor.grad_arrays 指向的是同一个对象
        # 。。。
        executor.arg_arrays = args
        executor.grad_arrays = args_grad
        executor.aux_arrays = aux_states
        return executor
```

* executor中存储的 `arg_arrays, grad_arrays, aux_arrays` 实际上是 `simple_bind` 的时候在 `Symbol` 对象中分配好的存储空间。



## Executor

```python
    def forward(self, is_train=False, **kwargs):
        """Calculate the outputs specified by the bound symbol.

        Parameters
        ----------
        is_train: bool, optional
            Whether this forward is for evaluation purpose. If True,
            a backward call is expected to follow. Otherwise following
            backward is invalid.

        **kwargs
            Additional specification of input arguments.

        Examples
        --------
        >>> # doing forward by specifying data
        >>> texec.forward(is_train=True, data=mydata)
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> mydata.copyto(texec.arg_dict['data'])
        >>> texec.forward(is_train=True)
        >>> # doing forward by specifying data and get outputs
        >>> outputs = texec.forward(is_train=True, data=mydata)
        >>> print(outputs[0].asnumpy())
        """
        if len(kwargs) != 0:
            arg_dict = self.arg_dict
            for name, array in kwargs.items():
                if not isinstance(array, (NDArray, np.ndarray)):
                    raise ValueError('only accept keyword argument of NDArrays and numpy.ndarray')
                if name not in arg_dict:
                    raise TypeError('Unknown argument %s' % name)
                if arg_dict[name].shape != array.shape:
                    raise ValueError('Shape not match! Argument %s, need: %s, received: %s'
                                     %(name, str(arg_dict[name].shape), str(array.shape)))
                # arg_array 还在原来的空间，并没有换空间。
                arg_dict[name][:] = array

        check_call(_LIB.MXExecutorForward(
            self.handle,
            ctypes.c_int(int(is_train))))

        if self._output_dirty:
            warnings.warn(
                "Calling forward the second time after forward(is_train=True) "
                "without calling backward first. Is this intended?", stacklevel=2)
        self._output_dirty = is_train

        return self.outputs 
```

* `self.arg_dict()`， `Variable` 的dict
* 将 用于输入的 `Variable` 的值给改了。注意，是把 `Variable`分配空间的值给改了，并不是指向了新的空间。
* 然后就开始 `forward` 了。 如何 `forward` 的不知。结果会返回，是个 `ndarray`

```python
    def backward(self, out_grads=None):
        """Do backward pass to get the gradient of arguments.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray or dict of str to NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.

        Examples
        --------
        >>> # Example for binding on loss function symbol, which gives the loss value of the model.
        >>> # Equivalently it gives the head gradient for backward pass.
        >>> # In this example the built-in SoftmaxOutput is used as loss function.
        >>> # MakeLoss can be used to define customized loss function symbol.
        >>> net = mx.sym.Variable('data')
        >>> net = mx.sym.FullyConnected(net, name='fc', num_hidden=6)
        >>> net = mx.sym.Activation(net, name='relu', act_type="relu")
        >>> net = mx.sym.SoftmaxOutput(net, name='softmax')

        >>> args =  {'data': mx.nd.ones((1, 4)), 'fc_weight': mx.nd.ones((6, 4)),
        >>>          'fc_bias': mx.nd.array((1, 4, 4, 4, 5, 6)), 'softmax_label': mx.nd.ones((1))}
        >>> args_grad = {'fc_weight': mx.nd.zeros((6, 4)), 'fc_bias': mx.nd.zeros((6))}
        >>> texec = net.bind(ctx=mx.cpu(), args=args, args_grad=args_grad)
        >>> out = texec.forward(is_train=True)[0].copy()
        >>> print out.asnumpy()
        [[ 0.00378404  0.07600445  0.07600445  0.07600445  0.20660152  0.5616011 ]]
        >>> texec.backward()
        >>> print(texec.grad_arrays[1].asnumpy())
        [[ 0.00378404  0.00378404  0.00378404  0.00378404]
         [-0.92399555 -0.92399555 -0.92399555 -0.92399555]
         [ 0.07600445  0.07600445  0.07600445  0.07600445]
         [ 0.07600445  0.07600445  0.07600445  0.07600445]
         [ 0.20660152  0.20660152  0.20660152  0.20660152]
         [ 0.5616011   0.5616011   0.5616011   0.5616011 ]]
        >>>
        >>> # Example for binding on non-loss function symbol.
        >>> # Here the binding symbol is neither built-in loss function
        >>> # nor customized loss created by MakeLoss.
        >>> # As a result the head gradient is not automatically provided.
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> # c is not a loss function symbol
        >>> c = 2 * a + b
        >>> args = {'a': mx.nd.array([1,2]), 'b':mx.nd.array([2,3])}
        >>> args_grad = {'a': mx.nd.zeros((2)), 'b': mx.nd.zeros((2))}
        >>> texec = c.bind(ctx=mx.cpu(), args=args, args_grad=args_grad)
        >>> out = texec.forward(is_train=True)[0].copy()
        >>> print(out.asnumpy())
        [ 4.  7.]
        >>> # out_grads is the head gradient in backward pass.
        >>> # Here we define 'c' as loss function.
        >>> # Then 'out' is passed as head gradient of backward pass.
        >>> texec.backward(out)
        >>> print(texec.grad_arrays[0].asnumpy())
        [ 8.  14.]
        >>> print(texec.grad_arrays[1].asnumpy())
        [ 4.  7.]
        """
        if out_grads is None:
            out_grads = []
        elif isinstance(out_grads, NDArray):
            out_grads = [out_grads]
        elif isinstance(out_grads, dict):
            out_grads = [out_grads[k] for k in self._symbol.list_outputs()]

        for obj in out_grads:
            if not isinstance(obj, NDArray):
                raise TypeError("inputs must be NDArray")
        ndarray = c_array(NDArrayHandle, [item.handle for item in out_grads])
        check_call(_LIB.MXExecutorBackward(
            self.handle,
            mx_uint(len(out_grads)),
            ndarray))

        if not self._output_dirty:
            warnings.warn(
                "Calling backward without calling forward(is_train=True) "
                "first. Behavior is undefined.", stacklevel=2)
        self._output_dirty = False
```

* 只需要这个函数跑完，`grad` 就乖乖的放到 `grad_arrays` 里了。存放的顺序和 `Symbol.list_arguments()` 的顺序一样。



**如何更新参数**

```python
# 一定要这么更新
for val, grad in zip(e.arg_arrays, e.grad_arrays):
    val -= grad

# 这样就血崩了。因为 executor 只任自家 存储空间的 东西。
for val, grad in zip(e.arg_arrays, e.grad_arrays):
    val =val -  grad
```



## 总结

可以这么理解 mxnet：

* Executor 是一个 类，里面保存着模型的 arguments （模型参数和 输入输出）
* feedforward 传入的参数改变了模型的输入输出，然后用Executor 计算
* backward 会求出所有 arguments 的梯度，就看你更不更新了。



**在使用 simple_bind 的时候**

* Symbol 对象中分配 了 模型 arguments（及其梯度）和 辅助参数的空间
* bind 方法使用这些 分配好的 空间（引用）
* bind 方法不会创建 新的空间来存 （arguments，。。。）




```python
# 测试代码
x = mx.sym.Variable("x")
y = x + 1
x_array = mx.nd.array([1., 2.], ctx=mx.cpu())
executor = y.bind(ctx=mx.cpu(), args=[x_array])
executor.forward()
print(executor.outputs[0].asnumpy())
x_array[1] = 3.
executor.forward()
print(executor.outputs[0].asnumpy())
```




## shape 推断 和 类型 推断

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

* infer_shape 的时候，`mxnet` 会推断出所有 `Variable` 的形状，和 Symbol 输出的形状和 辅助参数的形状。



## 遗留问题

* 如何不求 输入的 梯度呢？ 不取回来就好了。

