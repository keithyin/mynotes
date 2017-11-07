# backward

```python
def backward(self, out_grad=None, retain_graph=False, train_mode=True):
    """Compute the gradients of this NDArray w.r.t variables.

    Parameters
    ----------
    out_grad : NDArray, optional
        Gradient with respect to head.
    retain_graph : bool, optional
        Whether to retain the computaion graph for another backward
        pass on the same graph. By default the computaion history
        is cleared.
    train_mode : bool, optional
        Whether to compute gradient for training or inference.
    """
    if out_grad is None:
        ograd_handles = [NDArrayHandle(0)]
    else:
        ograd_handles = [out_grad.handle]

    check_call(_LIB.MXAutogradBackwardEx(
        1, c_array(NDArrayHandle, [self.handle]),
        c_array(NDArrayHandle, ograd_handles),
        0,
        ctypes.c_void_p(0),
        ctypes.c_int(retain_graph),
        ctypes.c_int(0),
        ctypes.c_int(train_mode),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0)))
```



```c
int MXAutogradBackwardEx(mx_uint num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         mx_uint num_variables,
                         NDArrayHandle *var_handles,
                         int retain_graph,
                         int create_graph,
                         int is_train,
                         NDArrayHandle **grad_handles,
                         int **grad_stypes) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();

  std::vector<NDArray*> outputs, ograds, variables;
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(reinterpret_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr) {
      ograds.emplace_back(reinterpret_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back(nullptr);
    }
  }

  variables.reserve(num_variables);
  // num_variables = 0
  for (mx_uint i = 0; i < num_variables; ++i) {
    variables.emplace_back(reinterpret_cast<NDArray*>(var_handles[i]));
  }
  // backward 是用 Imperative 执行的 这个类才是大佬
  auto grads = Imperative::Get()->Backward(outputs, ograds, variables, is_train,
                                                  retain_graph, create_graph);
  if (num_variables != 0) {
    ret->ret_handles.clear();
    ret->out_types.clear();
    ret->ret_handles.reserve(grads.size());
    ret->out_types.reserve(grads.size());
    for (const auto& i : grads) {
      ret->ret_handles.push_back(i);
      ret->out_types.push_back(i->storage_type());
    }
    *grad_handles = dmlc::BeginPtr(ret->ret_handles);
    *grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
  API_END();
}
```





## 参考资料

[https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc](https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc)

[https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc#L301](https://github.com/apache/incubator-mxnet/blob/master/src/c_api/c_api_ndarray.cc#L301)

