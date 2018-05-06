# pytorch cuda 语义
本篇博文是对[http://pytorch.org/docs/notes/cuda.html](http://pytorch.org/docs/notes/cuda.html)的理解性翻译，如有错误，请不吝指正！！


1. `pytorch`的`Tensor`分为两种，(i)`cpu tensor`(ii)`gpu tensor`。
2. `torch.cuda`中保存了当前选择的`gpu`
3. `torch.cuda.device()` 是一个上下文管理器，在这个上下文管理器中创建的`gpu tensor`会自动放在当前的`gpu`上。`torch.cuda.device(n)`可用来改变`torch.cuda`中保存的`gpu`。
4. `tensor`的计算结果和`tensor`会放置在同一个`gpu`上
5.  `gpu`之间的运算是不允许的，会报错，`copy_()`除外。

下面代码展示了`tensor`的`gpu`分配
```python
x = torch.cuda.FloatTensor(1)
# x.get_device() == 0
y = torch.FloatTensor(1).cuda()
# y.get_device() == 0

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.cuda.FloatTensor(1)

    # transfers a tensor from CPU to GPU 1
    b = torch.FloatTensor(1).cuda()
    # a.get_device() == b.get_device() == 1

    c = a + b
    # c.get_device() == 1

    z = x + y
    # z.get_device() == 0

    # even within a context, you can give a GPU id to the .cuda call
    d = torch.randn(2).cuda(2)
    # d.get_device() == 2
```
## 关于pinned memory
先看第一个程序
```python
import torch
value = torch.FloatTensor(3)

pined_value = value.pin_memory()

cuda_value = pined_value.cuda(async=True)

cuda_value.add_(torch.cuda.FloatTensor(3))
print(value)
print(pined_value)
print(cuda_value)
```
```
2.5166e-26
4.5567e-41
2.5166e-26
[torch.FloatTensor of size 3]


2.5166e-26
4.5567e-41
2.5166e-26
[torch.FloatTensor of size 3]


4.3844e-26
1.2578e-40
5.0332e-26
[torch.cuda.FloatTensor of size 3 (GPU 0)]
```
**可以看出 `async=True`时，当`GPU`上的`value`改变时是不会修改`CPU`上对应的值的。**

```python
import torch
value = torch.FloatTensor(3)

pined_value = value.pin_memory()

cuda_value = pined_value.cuda(async=False)

cuda_value.add_(torch.cuda.FloatTensor(3))

print(value)
print(pined_value)
print(cuda_value)
```
```
2.5166e-26
4.5567e-41
1.6809e-22
[torch.FloatTensor of size 3]


2.5166e-26
4.5567e-41
1.6809e-22
[torch.FloatTensor of size 3]


9.4176e-26
2.1692e-40
1.6816e-22
[torch.cuda.FloatTensor of size 3 (GPU 0)]
```
**可以看出 `async=False`时，当`GPU`上的`value`改变时`CPU`上对应的值不会进行同步。**

```python
import torch
value = torch.FloatTensor(3)

pined_value = value.pin_memory()

cuda_value = pined_value.cuda(async=False)

value.add_(torch.FloatTensor(3))

print(value)
print(pined_value)
print(cuda_value)
```
```
5.0332e-26
9.1135e-41
5.4845e-14
[torch.FloatTensor of size 3]


2.5166e-26
4.5567e-41
2.5166e-26
[torch.FloatTensor of size 3]


2.5166e-26
4.5567e-41
2.5166e-26
[torch.cuda.FloatTensor of size 3 (GPU 0)]
```
**可以看出，value值的改变不会改变pined_value, 也不会改变cuda_value的值**
