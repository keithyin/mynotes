# pytorch detach 与 detach_

pytorch 的 Variable 对象中有两个方法，detach和 detach_ 本文主要介绍这两个方法的效果和 能用这两个方法干什么。

## detach

官方文档中，对这个方法是这么介绍的。

* 返回一个新的 从当前图中分离的 Variable。
* 返回的 Variable 永远不会需要梯度
* 如果 被 detach 的Variable  volatile=True， 那么 detach 出来的 volatile 也为 True
* 还有一个注意事项，即：返回的 Variable 和 被 detach 的Variable 指向同一个 tensor


```python
import torch
from torch.nn import init
from torch.autograd import Variable
t1 = torch.FloatTensor([1., 2.])
v1 = Variable(t1)
t2 = torch.FloatTensor([2., 3.])
v2 = Variable(t2)
v3 = v1 + v2
v3_detached = v3.detach()
v3_detached.data.add_(t1) # 修改了 v3_detached Variable中 tensor 的值
print(v3, v3_detached)    # v3 中tensor 的值也会改变
```



```python
# detach 的源码
def detach(self):
    result = NoGrad()(self)  # this is needed, because it merges version counters
    result._grad_fn = None
    return result
```



## detach_

官网给的解释是：将 Variable 从创建它的 graph 中分离，把它作为叶子节点。

从源码中也可以看出这一点

* 将 Variable 的grad_fn 设置为 None，这样，BP 的时候，到这个 Variable 就找不到 它的 grad_fn，所以就不会再往后BP了。
* 将 requires_grad 设置为 False。这个感觉大可不必，但是既然源码中这么写了，如果有需要梯度的话可以再手动 将 requires_grad 设置为 true

```python
# detach_ 的源码
def detach_(self):
    """Detaches the Variable from the graph that created it, making it a
    leaf.
    """
    self._grad_fn = None
    self.requires_grad = False
```



## 能用来干啥

如果我们有两个网络 $A, B$, 两个关系是这样的 $y=A(x),  z = B(y)$  现在我们想用 $z.backward()$ 来为 $B$ 网络的参数来求梯度，但是又不想求 $A$ 网络参数的梯度。我们可以这样：

```python
# 第一种方法
y = A(x)
z = B(y.detach())
z.backward()

# 第二种方法
y = A(x)
y.detach_()
z = B(y)
z.backward()
```

在这种情况下，`detach 和 detach_` 都可以用。但是如果 你也想用 $y$ 来对 $A$ 进行 BP 呢？那就只能用第一种方法了。因为 第二种方法 已经将 $A$ 模型的输出  给 detach（分离）了。