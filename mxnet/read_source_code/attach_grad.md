# attach_grad

```python
def attach_grad(self, grad_req='write', stype=None):
    """Attach a gradient buffer to this NDArray, so that `backward`
    can compute gradient with respect to it.

    Parameters
    ----------
    grad_req : {'write', 'add', 'null'}
        How gradient will be accumulated.
        - 'write': gradient will be overwritten on every backward.
        - 'add': gradient will be added to existing value on every backward.
        - 'null': do not compute gradient for this NDArray.
    stype : str, optional
        The storage type of the gradient array. Defaults to the same stype of this NDArray.
    """
    from . import zeros as _zeros
    if stype is not None:
        grad = _zeros(self.shape, stype=stype)
    else:
        grad = op.zeros_like(self)  # pylint: disable=undefined-variable
    grad_req = _GRAD_REQ_MAP[grad_req]
    check_call(_LIB.MXAutogradMarkVariables(
        1, ctypes.pointer(self.handle),
        ctypes.pointer(mx_uint(grad_req)),
        ctypes.pointer(grad.handle)))
```



```python
_GRAD_REQ_MAP = {
    'null': 0,
    'write': 1,
    'add': 3
}
```





```c
int MXAutogradMarkVariables(mx_uint num_var,
                            NDArrayHandle *var_handles,
                            mx_uint *reqs_array,
                            NDArrayHandle *grad_handles) {
  API_BEGIN();
  std::vector<NDArray*> variables, gradients;
  std::vector<mx_uint> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (mx_uint i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  Imperative::Get()->MarkVariables(variables, grad_reqs, gradients);
  API_END();
}
```

