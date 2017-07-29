# module

## Module.Module()

> 初始化函数

* 会将 输入`Variable` 和 网络 `parameter` 分开

```python
data_names = list(data_names) if data_names is not None else []
label_names = list(label_names) if label_names is not None else []
state_names = list(state_names) if state_names is not None else []
fixed_param_names = list(fixed_param_names) if fixed_param_names is not None else []

arg_names = symbol.list_arguments()
input_names = data_names + label_names + state_names
self._param_names = [x for x in arg_names if x not in input_names]
self._fixed_param_names = fixed_param_names
self._aux_names = symbol.list_auxiliary_states()
self._data_names = data_names
self._label_names = label_names
self._state_names = state_names
self._output_names = symbol.list_outputs()
```



## Module.bind

```python
# 会创建一个这个对象
self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                     self._work_load_list, 																	 self._data_shapes,
                                                     self._label_shapes, 																	 self._param_names,
                                                     for_training, inputs_need_grad,
                                                     shared_group, logger=self.logger,
                                               fixed_param_names=self._fixed_param_names,
                                                     grad_req=grad_req,
                                                     state_names=self._state_names)
```



## Executor 的初始化函数

```python
    def __init__(self, symbol, contexts, workload, data_shapes, label_shapes, param_names,
                 for_training, inputs_need_grad, shared_group=None, logger=logging,
                 fixed_param_names=None, grad_req='write', state_names=None):
        self.param_names = param_names
        self.arg_names = symbol.list_arguments()
        self.aux_names = symbol.list_auxiliary_states()

        self.symbol = symbol
        self.contexts = contexts
        self.workload = workload

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad

        self.logger = logger
        #In the future we should have a better way to profile memory per device (haibin)
        self._total_exec_bytes = 0
        self.fixed_param_names = fixed_param_names
        if self.fixed_param_names is None:
            self.fixed_param_names = []

        self.state_names = state_names
        if self.state_names is None:
            self.state_names = []

        if not for_training:
            grad_req = 'null'

        data_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
        if label_shapes is not None:
            label_shapes = [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]

        data_names = [x.name for x in data_shapes]

        if isinstance(grad_req, str):
            self.grad_req = {}
            for k in self.arg_names:
                if k in self.param_names:
                    self.grad_req[k] = 'null' if k in self.fixed_param_names else grad_req
                elif k in data_names:
                    self.grad_req[k] = grad_req if self.inputs_need_grad else 'null'
                else:
                    self.grad_req[k] = 'null'
        elif isinstance(grad_req, (list, tuple)):
            assert len(grad_req) == len(self.arg_names)
            self.grad_req = dict(zip(self.arg_names, grad_req))
        elif isinstance(grad_req, dict):
            self.grad_req = {}
            for k in self.arg_names:
                if k in self.param_names:
                    self.grad_req[k] = 'null' if k in self.fixed_param_names else 'write'
                elif k in data_names:
                    self.grad_req[k] = 'write' if self.inputs_need_grad else 'null'
                else:
                    self.grad_req[k] = 'null'
            self.grad_req.update(grad_req)
        else:
            raise ValueError("grad_req must be one of str, list, tuple, or dict.")

        if shared_group is not None:
            self.shared_data_arrays = shared_group.shared_data_arrays
        else:
            self.shared_data_arrays = [{} for _ in contexts]

        # initialize some instance variables
        self.batch_size = None
        self.slices = None
        self.execs = []
        self._default_execs = None
        self.data_arrays = None
        self.label_arrays = None
        self.param_arrays = None
        self.state_arrays = None
        self.grad_arrays = None
        self.aux_arrays = None
        self.input_grad_arrays = None

        self.data_shapes = None
        self.label_shapes = None
        self.data_names = None
        self.label_names = None
        self.data_layouts = None
        self.label_layouts = None
        self.output_names = self.symbol.list_outputs()
        self.output_layouts = [DataDesc.get_batch_axis(self.symbol[name].attr('__layout__'))
                               for name in self.output_names]
        self.num_outputs = len(self.symbol.list_outputs())

        self.bind_exec(data_shapes, label_shapes, shared_group)
```

```python
    # 这里也没有分配任何空间
    def bind_exec(self, data_shapes, label_shapes, shared_group=None, reshape=False):
        """Bind executors on their respective devices.

        Parameters
        ----------
        data_shapes : list
        label_shapes : list
        shared_group : DataParallelExecutorGroup
        reshape : bool
        """
        assert reshape or not self.execs
        self.batch_size = None

        # calculate workload and bind executors
        self.data_layouts = self.decide_slices(data_shapes)
        if label_shapes is not None:
            # call it to make sure labels has the same batch size as data
            self.label_layouts = self.decide_slices(label_shapes)

        for i in range(len(self.contexts)):
            data_shapes_i = self._sliced_shape(data_shapes, i, self.data_layouts)
            if label_shapes is not None:
                label_shapes_i = self._sliced_shape(label_shapes, i, self.label_layouts)
            else:
                label_shapes_i = []

            if reshape:
                self.execs[i] = self._default_execs[i].reshape(
                    allow_up_sizing=True, **dict(data_shapes_i + label_shapes_i))
            else:
                self.execs.append(self._bind_ith_exec(i, data_shapes_i, label_shapes_i,
                                                      shared_group))

        self.data_shapes = data_shapes
        self.label_shapes = label_shapes
        self.data_names = [i.name for i in self.data_shapes]
        if label_shapes is not None:
            self.label_names = [i.name for i in self.label_shapes]
        self._collect_arrays()
```



```python
    # 这儿可以看出是在 exec 中分配的存储空间。
    def _collect_arrays(self):
        """Collect internal arrays from executors."""
        # convenient data structures
        self.data_arrays = [[(self.slices[i], e.arg_dict[name]) for i, e in enumerate(self.execs)]
                            for name, _ in self.data_shapes]

        self.state_arrays = [[e.arg_dict[name] for e in self.execs]
                             for name in self.state_names]

        if self.label_shapes is not None:
            self.label_arrays = [[(self.slices[i], e.arg_dict[name])
                                  for i, e in enumerate(self.execs)]
                                 for name, _ in self.label_shapes]
        else:
            self.label_arrays = None

        self.param_arrays = [[exec_.arg_arrays[i] for exec_ in self.execs]
                             for i, name in enumerate(self.arg_names)
                             if name in self.param_names]
        if self.for_training:
            self.grad_arrays = [[exec_.grad_arrays[i] for exec_ in self.execs]
                                for i, name in enumerate(self.arg_names)
                                if name in self.param_names]
        else:
            self.grad_arrays = None

        data_names = [x[0] for x in self.data_shapes]
        if self.inputs_need_grad:
            self.input_grad_arrays = [[exec_.grad_arrays[self.arg_names.index(name)]
                                       for exec_ in self.execs]
                                      for name in data_names if name in self.arg_names]
        else:
            self.input_grad_arrays = None

        self.aux_arrays = [[exec_.aux_arrays[i] for exec_ in self.execs]
                           for i in range(len(self.aux_names))]
```



```python
# 是在 ExecutorGroup 的 def _bind_ith_exec 中 给所有的 `Variable` 分配的空间。
def _bind_ith_exec()
```

