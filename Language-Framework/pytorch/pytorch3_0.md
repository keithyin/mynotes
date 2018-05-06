# pytorch学习笔记（3）：模块介绍

在`pytorch`中，神经网络的创建和`torch.nn`这个包是分不开的。
## torch.nn 包
包含神经网络的基本模块：卷积，全连接

## torch.nn.functional
包含了激活函数，也包含 `pooling`，不包含参数的函数的都在这里

## torch.optim
包含优化器， `SGD`， `ada`都有。
`optim`帮助我们管理更新，不用手动一个个更新了。
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

## torch.nn.Module
`pytorch`中，神经网络的创建，首先要定义一个类，这个类需要继承`nn.Module`。`nn.Module`可以帮助我们管理网络中的参数。也可以很容易的将参数放到`gpu`上。不用一个个网上放了。
`pytorch`编程框架：
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):#必须要实现的，否则报错，通过net(x),就可以调用这个函数，因为和call绑定了
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
从这个示例代码也可以看出，`pytorch`的编程框架是：
```python
- 定义网络结构
- 创建对象的时候，初始化网络参数
for i in range(steps):
  - 执行 forward()
  - 计算 loss()
  - 更新参数
```
