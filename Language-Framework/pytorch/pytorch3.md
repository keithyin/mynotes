
#  auto gradient
本片博文主要是对[http://pytorch.org/docs/notes/autograd.html](http://pytorch.org/docs/notes/autograd.html)的部分翻译以及自己的理解，如有错误，欢迎指正！

## Backward过程中排除子图
`pytorch`的`BP`过程是由一个函数决定的，`loss.backward()`， 可以看到`backward()`函数里并没有传要求谁的梯度。那么我们可以大胆猜测，在`BP`的过程中，`pytorch`是将所有影响`loss`的`Variable`都求了一次梯度。**但是有时候，我们并不想求所有`Variable`的梯度。**那就要考虑如何`在Backward过程中排除子图`（ie.排除没必要的梯度计算）。
如何`BP`过程中排除子图？ `Variable`的两个参数（`requires_grad`和`volatile`）

`requires_grad`: 


```python
import torch
from torch.autograd import Variable
x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5))
z = Variable(torch.randn(5, 5), requires_grad=True)
a = x + y  # x, y的 requires_grad的标记都为false， 所以输出的变量requires_grad也为false
a.requires_grad
```




    False




```python
b = a + z #a ,z 中，有一个 requires_grad 的标记为True，那么输出的变量的 requires_grad为True
b.requires_grad
```




    True



变量的`requires_grad`标记的运算就相当于`or`。
如果你想部分冻结你的网络（ie.不做梯度计算），那么通过设置`requires_grad`标签是非常容易实现的。
下面给出了利用`requires_grad`使用`pretrained`网络的一个例子，只`fine tune`了最后一层。


```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
```

`volatile`：


```python
j = Variable(torch.randn(5,5), volatile=True)
k = Variable(torch.randn(5,5))
m = Variable(torch.randn(5,5))
n = k+m # k,m变量的volatile标记都为False，输出的Variable的volatile标记也为false
n.volatile
```




    False




```python
o = j+k #k,m变量的volatile标记有一个True，输出的Variable的volatile为True
o.volatile
```




    True



变量的`volatile`标记的运算也相当于`or`。
注意：`volatile=True`相当于`requires_grad=False`。但是在纯推断模式的时候，只要是输入`volatile=True`，那么输出Variable的`volatile`必为`True`。这就比使用`requires_grad=False`方便多了。

`NOTE`：**在使用`volatile=True`的时候，变量是不存储 `creator`属性的，这样也减少了内存的使用。**

## 为什么要排除子图
也许有人会问，梯度全部计算，不更新的话不就得了。
这样就涉及了效率的问题了，计算很多没用的梯度是浪费了很多资源的（时间，计算机内存）

## 参考资料
[http://pytorch.org/docs/notes/autograd.html](http://pytorch.org/docs/notes/autograd.html)
