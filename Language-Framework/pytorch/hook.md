# pytorch hook

## Module

* `register_forward_hook()`
* `register_forward_pre_hook()`
* `register_backward_hook()`

```python
def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, input) # hook 不应该修改 input 的值
        if torch.jit._tracing:
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result) # forward hook 用在这里。
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result
```



**总结**

* `forward_hook` : 不会对正常的 计算图产生影响，只是对 `Module` 的输入和输出进行窥探。
* `register_backward_hook()`: 是注册到 `Module` 输出的 `Tensor` 的 `grad_fn` 上的。**可以用来影响梯度流**

```python
import torch
from torch import nn

# hook 返回的值作为最终的 grad_in
# grad_in 表示前向时候的输入的梯度
# grad_out 表示前向的时候的输出的梯度
def hook(module, grad_in, grad_out):
    res = (torch.ones_like(grad) for grad in grad_in)
    return tuple(res)


if __name__ == '__main__':
    linear = nn.Linear(10, 20)

    linear.register_backward_hook(hook)

    a = torch.randn(3, 10, requires_grad=True)

    res = torch.sum(linear(a))
    res.backward()
    print(a.grad)
    
"""
tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
"""
```





## Tensor

* `register_hook()`： 对自己的梯度处理一下。处理后的作为自身的最终梯度。



