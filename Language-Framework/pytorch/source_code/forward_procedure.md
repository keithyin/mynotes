# pytorch 的前向计算过程



这个是 `Module` 中 `__call__` 方法，定义了 神经网络的前向传导过程！

```python
def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
        hook(self, input)
    result = self.forward(*input, **kwargs)
    for hook in self._forward_hooks.values():
        hook_result = hook(self, input, result)
        if hook_result is not None:
            raise RuntimeError(
                "forward hooks should never return any values, but '{}'"
                "didn't return None".format(hook))
    if len(self._backward_hooks) > 0:
        var = result
        while not isinstance(var, Variable):
            var = var[0]
        grad_fn = var.grad_fn
        if grad_fn is not None:
            for hook in self._backward_hooks.values():
                # 将 hook 的一个参数 设置成 module, 加个 wrapper
                wrapper = functools.partial(hook, self)
                functools.update_wrapper(wrapper, hook)
                grad_fn.register_hook(wrapper)
    return result
```



**基本流程为**

* `pre_hook` 瞅一瞅 输入是啥，只可远看，不可亵玩。
* 执行 `forward` ，走一波 前向传到过程
* `forward_hook` 瞅一瞅输出是啥，只可远看，不可亵玩
* 给 `grad_fn` 注册一波 `backward_hook`, 每次前向调用都会重新注册，因为 `grad_fn` 每次都会重新创建！





## 关于 functools.partial

给一个函数的某个位置 指定好参数。其实就是 远函数 加了个 wrapper。

```python
import functools

def demo_1(name, age):
    print(name, age)

# 第一个位置的参数 为 3
res_func = functools.partial(demo_1, 3)
print(res_func(2))
```



```python
def partial(func, *args, **keywords):
    """New function with partial application of the given arguments
    and keywords.
    """
    if hasattr(func, 'func'):
        args = func.args + args
        tmpkw = func.keywords.copy()
        tmpkw.update(keywords)
        keywords = tmpkw
        del tmpkw
        func = func.func

    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        # 有 key 则更新， 没有 key 则添加
        newkeywords.update(fkeywords)
        # newfunc 是一个 wrapper，里面包装着之前的 func
        return func(*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
```





