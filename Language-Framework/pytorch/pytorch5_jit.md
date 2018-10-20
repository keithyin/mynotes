# pytorch1.0 特性-jit

[https://pytorch.org/blog/the-road-to-1_0/](https://pytorch.org/blog/the-road-to-1_0/)

为什么使用 jit?

* 减少与python的交互, 因为python相比C/C++还是比较慢的.
* 使用 jit 还可以对计算图进行优化



两种方式:

* tracing
* scripting



## Tracing

* 如果函数或者可调用对象中**没有控制流** ,使用 tracing

```python
import torch

def foo(x, y):
    return 2 * x + y

# 用在没有控制流的函数中
traced_foo = torch.jit.trace(foo, (torch.tensor(3.), torch.tensor(2.)))


# 用在没有控制流的可调用对象上
conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
```



## Script

* 如果函数中有控制流, 使用 script

```python
import torch
@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r
```



## ScriptModule

* 如果要创建一个带有参数的 Module 的话,需要这么操作

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import ScriptModule, script_method, trace

class MyScriptModule(ScriptModule):
    def __init__(self):
        super(MyScriptModule, self).__init__()
        # trace produces a ScriptModule's conv1 and conv2
        self.conv1 = trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
        self.conv2 = trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))
		self.weight = torch.nn.Parameter(torch.rand(N, M))
    @script_method
    def forward(self, input):
      input = F.relu(self.conv1(input))
      input = F.relu(self.conv2(input))
      input = self.weight.mv(input)
      return input
```





## 纠结的地方

C++ 如何动态构建对象.