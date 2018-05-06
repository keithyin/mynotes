
# gradient
在`BP`的时候，`pytorch`是将`Variable`的梯度放在`Variable`对象中的，我们随时都可以使用`Variable.grad`得到对应`Variable`的`grad`。刚创建`Variable`的时候，它的`grad`属性是初始化为`0.0`的。


```python
import torch
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
print(w1.grad)
print(w2.grad)
```

    Variable containing:
     0
     0
     0
    [torch.FloatTensor of size 3]

    Variable containing:
     0
     0
     0
    [torch.FloatTensor of size 3]



**从下面这两段代码可以看出，`Variable`的`grad`是累加的即: `Variable.grad=Variable.grad+new_grad`**


```python
d = torch.mean(w1)
d.backward()
w1.grad
```




    Variable containing:
     0.3333
     0.3333
     0.3333
    [torch.FloatTensor of size 3]




```python
d.backward()
w1.grad
```




    Variable containing:
     0.6667
     0.6667
     0.6667
    [torch.FloatTensor of size 3]



**既然累加的话，那我们如何置零呢？**


```python
w1.grad.data.zero_()
w1.grad
```




    Variable containing:
     0
     0
     0
    [torch.FloatTensor of size 3]



通过上面的方法，就可以将`grad`置零。通过打印出来的信息可以看出，`w1.grad`其实是`Tensor`。现在可以更清楚的理解一下`Variable`与`Tensor`之间的关系，上篇博客已经说过，`Variable`是`Tensor`的一个`wrapper`，那么到底是什么样的`wrapper`呢？从目前的掌握的知识来看，`Variable`中至少包含了两个`Tensor`，一个是保存`weights`的`Tensor`，一个是保存`grad`的`Tensor`。`Variable`的一些运算，实际上就是里面的`Tensor`的运算。
`pytorch`中的所有运算都是基于`Tensor`的，`Variable`只是一个`Wrapper`，`Variable`的计算的实质就是里面的`Tensor`在计算。`Variable`默认代表的是里面存储的`Tensor`（`weights`）。理解到这，我们就可以对`grad`进行随意操作了。


```python
# 获得梯度后，如何更新
learning_rate = 0.1
w1.data -= learning_rate * w1.grad  # w1.data是获取保存weights的Tensor
```

## torch.optim
如果每个参数的更新都要`w1.data.sub_(learning_rate*w1.grad.data)`，那就比较头疼了。还好，`pytorch`为我们提供了`torch.optim`包，这个包可以简化我们更新参数的操作。
```python
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
for i in range(steps):
  optimizer.zero_grad() # zero the gradient buffers，必须要置零
  output = net(input)
  loss = criterion(output, target)
  loss.backward()
  optimizer.step() # Does the update
```
注意：`torch.optim`只用于更新参数，不care梯度的计算。
