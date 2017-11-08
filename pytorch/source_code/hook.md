# pytorch 的 hook 机制


在看`pytorch`官方文档的时候，发现在`nn.Module`部分和`Variable`部分均有`hook`的身影。感到很神奇，因为在使用`tensorflow`的时候没有碰到过这个词。所以打算一探究竟。

## Variable 的 hook

* `register_hook(hook)`

### register_hook(hook)
注册一个`backward`钩子。

每次`gradients`被计算的时候，这个`hook`都被调用。`hook`应该拥有以下签名：

`hook(grad) -> Variable or None`

`hook`不应该修改它的输入，但是它可以返回一个替代当前梯度的新梯度。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

例子：
```python
v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
h = v.register_hook(lambda grad: grad * 2)  # double the gradient
v.backward(torch.Tensor([1, 1, 1]))
#先计算原始梯度，再进hook，获得一个新梯度。
print(v.grad.data)
h.remove()  # removes the hook，这个是调用 RemovableHandle 的 remove 函数
```

```
 2
 2
 2
[torch.FloatTensor of size 3]
```

**这里看一下， Variable 中的源码**

```python
def register_hook(self, hook):
    if self.volatile:
        raise RuntimeError("cannot register a hook on a volatile variable")
    if not self.requires_grad:
        raise RuntimeError("cannot register a hook on a variable that "
                           "doesn't require gradient")
    # Variable 类中维护了一个 _backward_hooks
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
        # 如果 grad_fn 不为 空的话， 把这个 Variable 注册到 grad_fn 上去
        if self.grad_fn is not None:
            self.grad_fn._register_hook_dict(self)
    # 每次调用这个函数 handle.id 都要加一
    # 有个问题，这里为什么每次都要创建一个 RemoableHandle 对象呢？
    handle = hooks.RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle
```



```python
class RemovableHandle(object):
    """A handle which provides the capability to remove a hook."""

    next_id = 0

    def __init__(self, hooks_dict):
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self):
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return (self.hooks_dict_ref(), self.id)

    def __setstate__(self, state):
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(collections.OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
```



**总结**

* 对 Variable 注册的 hook 都会存在 `Variable._backward_hooks` 这个 字典中， `hook_handle.remove` 就是将这个 `_backward_hooks` 中的注册的最后一个 `hook` 删除掉
* 如果 `Variable.grad_fn` 不为 `None`，那么 此 `Variable` 会注册到 `grad_fn` 上，这是要搞什么呢？



## nn.Module的hook

* `register_forward_hook(hook)`
* `register_forward_pre_hook(hook)`
* `register_backward_hook(hook)`

###  register_forward_pre_hook

```python
def register_forward_pre_hook(self, hook):
    """Registers a forward pre-hook on the module.

    The hook will be called before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None

    The hook should not modify the input.
    This function returns a handle with a method ``handle.remove()``
    that removes the hook from the module.
    """
    handle = hooks.RemovableHandle(self._forward_pre_hooks)
    self._forward_pre_hooks[handle.id] = hook
    return handle
```



### register_forward_hook(hook)
在`module`上注册一个`forward hook`。
> 这里要注意的是，hook 只能注册到 Module 上，即，仅仅是简单的 `op` 包装的 Module，而不是我们继承 Module时写的那个类，我们继承 Module写的类叫做 Container。
> 每次调用`forward()`计算输出的时候，这个`hook`就会被调用。它应该拥有以下签名：

`hook(module, input, output) -> None`

`hook`不应该修改 `input`和`output`的值。 这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

看这个解释可能有点蒙逼，但是如果要看一下`nn.Module`的源码怎么使用`hook`的话，那就乌云尽散了。
先看 `register_forward_hook`
```python
def register_forward_hook(self, hook):

    handle = hooks.RemovableHandle(self._forward_hooks)
    self._forward_hooks[handle.id] = hook
    return handle
```
这个方法的作用是在此`module`上注册一个`hook`，函数中第一句就没必要在意了，主要看第二句，是把注册的`hook`保存在`_forward_hooks`字典里。

再看 `nn.Module` 的`__call__`方法（被阉割了，只留下需要关注的部分）：

```python

def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
        hook(self, input)
    result = self.forward(*input, **kwargs)
    for hook in self._forward_hooks.values():
       #将注册的hook拿出来用
        hook_result = hook(self, input, result)
    if len(self._backward_hooks) > 0:
        var = result
        while not isinstance(var, Variable):
            var = var[0]
        grad_fn = var.grad_fn
        if grad_fn is not None:
            for hook in self._backward_hooks.values():
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                grad_fn.register_hook(wrapper)
    return result
```
可以看到，当我们执行`model(x)`的时候，底层干了以下几件事：

- 调用 `forward` 方法计算结果

- 判断有没有注册 `forward_hook`，有的话，就将 `forward` 的输入及结果作为`hook`的实参。然后让`hook`自己干一些不可告人的事情。

看到这，我们就明白`hook`签名的意思了，还有为什么`hook`不能修改`input`的`output`的原因。

小例子：
```python
import torch
from torch import nn
import torch.functional as F
from torch.autograd import Variable

def for_hook(module, input, output):
    print(module)
    for val in input:
        print("input val:",val)
    for out_val in output:
        print("output val:", out_val)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):

        return x+1

model = Model()
x = Variable(torch.FloatTensor([1]), requires_grad=True)
handle = model.register_forward_hook(for_hook)
print(model(x))
handle.remove()
```


### register_backward_hook

在`module`上注册一个`bachward hook`。**此方法目前只能用在`Module`上，不能用在`Container`上，当`Module`的forward函数中只有一个`Function`的时候，称为`Module`，如果`Module`包含其它`Module`，称之为`Container`**

每次计算`module`的`inputs`的梯度的时候，这个`hook`会被调用。`hook`应该拥有下面的`signature`。

`hook(module, grad_input, grad_output) -> Tensor or None`

如果`module`有多个输入输出的话，那么`grad_input` `grad_output`将会是个`tuple`。
`hook`不应该修改它的`arguments`，但是它可以选择性的返回关于输入的梯度，这个返回的梯度在后续的计算中会替代`grad_input`。

这个函数返回一个 句柄(`handle`)。它有一个方法 `handle.remove()`，可以用这个方法将`hook`从`module`移除。

从上边描述来看，`backward hook`似乎可以帮助我们处理一下计算完的梯度。看下面`nn.Module`中`register_backward_hook`方法的实现，和`register_forward_hook`方法的实现几乎一样，都是用字典把注册的`hook`保存起来。
```python
def register_backward_hook(self, hook):
    handle = hooks.RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle
```
先看个例子来看一下`hook`的参数代表了什么：
```python
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import math
def bh(m,gi,go):
    print("Grad Input")
    print(gi)
    print("Grad Output")
    print(go)
    return gi[0]*0,gi[1]*0
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is None:
            return self._backend.Linear()(input, self.weight)
        else:
            return self._backend.Linear()(input, self.weight, self.bias)

x=Variable(torch.FloatTensor([[1, 2, 3]]),requires_grad=True)
mod=Linear(3, 1, bias=False)
mod.register_backward_hook(bh) # 在这里给module注册了backward hook

out=mod(x)
out.register_hook(lambda grad: 0.1*grad) #在这里给variable注册了 hook
out.backward()
print(['*']*20)
print("x.grad", x.grad)
print(mod.weight.grad)
```
```
Grad Input
(Variable containing:
1.00000e-02 *
  5.1902 -2.3778 -4.4071
[torch.FloatTensor of size 1x3]
, Variable containing:
 0.1000  0.2000  0.3000
[torch.FloatTensor of size 1x3]
)
Grad Output
(Variable containing:
 0.1000
[torch.FloatTensor of size 1x1]
,)
['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']
x.grad Variable containing:
 0 -0 -0
[torch.FloatTensor of size 1x3]

Variable containing:
 0  0  0
[torch.FloatTensor of size 1x3]
```
可以看出，`grad_in`保存的是，此模块`Function`方法的输入的值的梯度。`grad_out`保存的是，此模块`forward`方法返回值的梯度。我们不能在`grad_in`上直接修改，但是我们可以返回一个新的`new_grad_in`作为`Function`方法`inputs`的梯度。

上述代码对`variable`和`module`同时注册了`backward hook`，这里要注意的是，无论是`module hook`还是`variable hook`，最终还是注册到`Function`上的。这点通过查看`Varible`的`register_hook`源码和`Module`的`__call__`源码得知。
> Module的register_backward_hook的行为在未来的几个版本可能会改变

BP过程中`Function`中的动作可能是这样的
```python
class Function:
	def __init__(self):
		...
	def forward(self, inputs):
		...
		return outputs
	def backward(self, grad_outs):
		...
		return grad_ins
	def _backward(self, grad_outs):
		hooked_grad_outs = grad_outs
		for hook in hook_in_outputs:
			hooked_grad_outs = hook(hooked_grad_outs)
		grad_ins = self.backward(hooked_grad_outs)
		hooked_grad_ins = grad_ins
		for hook in hooks_in_module:
			hooked_grad_ins = hook(hooked_grad_ins)
		return hooked_grad_ins
```
关于`pytorch run_backward()`的可能实现猜测为。
```python
def run_backward(variable, gradient):
	creator = variable.creator
	if creator is None:
		variable.grad = variable.hook(gradient)
		return 
	grad_ins = creator._backward(gradient)
	vars = creator.saved_variables
	for var, grad in zip(vars, grad_ins):
		run_backward(var, var.grad)
```





**中间Variable的梯度在BP的过程中是保存到GradBuffer中的(C++源码中可以看到), BP完会释放. 如果retain_grads=True的话,就不会被释放**



## 总结

