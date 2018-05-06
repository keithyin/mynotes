# pytorch学习笔记（十七）：扩展 pytorch-python版

`pytorch` 虽然提供了很多的 `op` 使得我们很容易的使用。但是当已有的 `op` 无法满足我们的要求的时候，那就需要自己动手来扩展。 `pytorch` 提供了两种方式来扩展 `pytorch` 的基础功能。 

* 通过继承 `autograd.Function`
* 通过 `C` 来扩展

本篇博客主要介绍 继承 `autograd.Function` 来扩展 `pytorch`。

[官方文档链接](http://pytorch.org/docs/0.3.0/notes/extending.html)



继承 `autograd.Function` 的 子类 只需要 实现两个 静态方法：

* `forward` ： 计算 `op` 的前向过程，官方文档中提到了几个点
  * 在执行 `forward` 之前，`Variable` 参数已经被转换成了 `Tensor`
  * `forward` 的形参可以有默认参数，默认参数可以是任意 `python` 对象。
  * 可以返回任意多个 `Tensor` 
* `backward` ：  计算 梯度，
  * `forward ` 返回几个 值， 这里就需要几个 形参，还得外加一个 `ctx`。
  * `forward` 有几个 形参（不包含 `ctx`） ，`backward` 就得返回几个值。



**一个 Demo（来自官网）**

```python
class LinearFunction(Function):
    # forward 和 backward 都得是 静态方法！！！！！
    @staticmethod
    # bias 是个可选参数，有个 默认值 None
    def forward(ctx, input, weight, bias=None):
        # input，weight 都已经变成了 Tensor
        # 用 ctx 把该存的存起来，留着 backward 的时候用
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # 由于 forward 只有一个 返回值，所以 backward 只需要一个参数 接收 梯度。
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是 Variable 类型。
        # 在开头的地方将保存的 tensor 给 unpack 了
        # 然后 给 所有应该返回的 梯度 以 None 初始化。
        # saved_variables 返回的是 Variable！！！ 不是 Tensor 了。
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # needs_input_grad 检查是可选的。如果想使得 代码更简单的话，可以忽略。
        # 给不需要梯度的 参数返回梯度 不是一个错误。
		# 返回值 的个数 需要和 forward 形参的个数（不包含 ctx）一致
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
		# 梯度的顺序和 forward 形参的顺序要对应。
        return grad_input, grad_weight, grad_bias
```

**关于 ctx**

* `save_for_backward` 只能存 **`tensor, None`**, 其余都不能存。
* `save_for_backward` 只保存 `forward` 的实参，或者 `forward` 的返回值。



**上面就是继承 `Function` 的全过程，然后该怎么使用呢？**

```python
# input, weight, 是 Variable
def linear(input, weight, bias=None):
    # 一定是要 通过调用 apply 来用的。 Function.apply 中估计做了不少事情。
    return LinearFunction.apply(input, weight, bias)
```



**也可以将 LinearFunction 封装到 `nn.Module` 里面，以便更简单的使用。**



## 检查梯度计算是否正确

`pytorch` 提供了一个简单的 接口用来检查 定义的 梯度计算是否正确

```python
from torch.autograd import gradcheck
# Check gradients computed via small finite differences against analytical gradients

# 检查的是 inputs 中 requires_grad=True 的梯度，
# 一定要记得 double() 一下！！！！！！
input = (Variable(torch.randn(20, 20).double(), requires_grad=True),
             Variable(torch.randn(30, 20).double(), requires_grad=True),)
test = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4)
# 如果通过，最后会打印一个 True
print(test)
```





## 总结

* `forward` 的形参是 `Tensor`， `return` 的也是 `Tensor`
* `backward` 的形参是  `Variable`， `return` 也需要是 `Variable`
* `gradcheck` 的时候，记得将 `Tensor` 的类型转成 `double`， 使用 `float` 会导致检查失败。
* ​



**GlobalMaxPool例子**

```python
class GlobalMaxPool(Function):
    @staticmethod
    def forward(ctx, inputs):
        bs, c, h, w = inputs.size()
        flatten_hw = inputs.view(bs, c, -1)
        max_val, indices = torch.max(flatten_hw, dim=-1, keepdim=True)
        max_val = max_val.view(bs, c, 1, 1)
        ctx.save_for_backward(inputs, indices)
        # 只有返回 indices， 才让 save_for_backward。。。 迫不得已。
        return max_val, indices

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_max_val, grad_indices):
        inputs, indices = ctx.saved_variables

        bs, c, h, w = inputs.size()
        grad_inputs = inputs.data.new().resize_as_(inputs.data).zero_().view(bs, c, -1)
        grad_inputs.scatter_(-1, indices.data,
                             torch.squeeze(grad_max_val.data).contiguous().view(bs, c, 1))
        grad_inputs = grad_inputs.view_as(inputs.data)

        return Variable(grad_inputs, volatile=grad_max_val.volatile)


def global_max_pool(input):
    return GlobalMaxPool.apply(input)


if __name__ == '__main__':
    in_ = Variable(torch.randn(2, 1, 3, 3).double(), requires_grad=True)
    res, _ = global_max_pool(in_)
    # print(res)

    res.sum().backward()
    res = gradcheck(GlobalMaxPool.apply, (in_,))
    print(res)
```

