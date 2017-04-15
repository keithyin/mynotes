
# pytorch 学习笔记(一)
`pytorch`是一个动态的建图的工具。不像`Tensorflow`那样，先建图，然后通过`feed`和`run`重复执行建好的图。相对来说，`pytorch`具有更好的灵活性。

编写一个深度网络需要关注的地方是：
1. 如何保存参数
2. 如何构建网络
3. 如何计算梯度和更新参数

## 如何保存参数
`pytorch`中有两种变量类型，一个是`Tensor`，一个是`Variable`。
- `Tensor`： 就像`ndarray`一样,多维向量
- `Variable`：是`Tensor`的一个`wrapper`，不仅保存了值，而且保存了这个值的`creator`


```python
import torch
x  = torch.Tensor(2,3,4) # torch.Tensor(shape) 创建出一个未初始化的Tensor，官方文档是这么说，但是打印出来的话还是有值的
x                        # 但是这种方式创建出来的Tensor更多是用来接受其他数据的计算值的
```




    
    (0 ,.,.) = 
    1.00000e-37 *
       1.5926  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
    
    (1 ,.,.) = 
    1.00000e-37 *
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
    [torch.FloatTensor of size 2x3x4]




```python
x.size()
```




    torch.Size([2, 3, 4])




```python
a = torch.rand(2,3,4)
b = torch.rand(2,3,4)
_=torch.add(a,b, out=x)  # 使用Tensor()方法创建出来的Tensor用来接收计算结果，当然torch.add(..)也会返回计算结果的
x
```




    
    (0 ,.,.) = 
      0.9815  0.0833  0.8217  1.1280
      0.7810  1.2586  1.0243  0.7924
      1.0200  1.0463  1.4997  1.0994
    
    (1 ,.,.) = 
      0.8031  1.4283  0.6245  0.9617
      1.3551  1.9094  0.9046  0.5543
      1.2838  1.7381  0.6934  0.8727
    [torch.FloatTensor of size 2x3x4]




```python
a.add_(b) # 所有带 _ 的operation，都会更改调用对象的值，
#例如 a=1;b=2; a.add_(b); a就是3了，没有 _ 的operation就没有这种效果，只会返回运算结果
torch.cuda.is_available()
```




    True



## 自动求导
`pytorch`的自动求导工具包在`torch.autograd`中


```python
from torch.autograd import Variable
x = torch.rand(5)
x = Variable(x,requires_grad = True)
y = x * 2
grads = torch.FloatTensor([1,2,3,4,5])
y.backward(grads)#如果y是scalar的话，那么直接y.backward()，然后通过x.grad方式，就可以得到var的梯度
x.grad           #如果y不是scalar，那么只能通过传参的方式给x指定梯度
```




    Variable containing:
      2
      4
      6
      8
     10
    [torch.FloatTensor of size 5]



## neural networks
使用`torch.nn`包中的工具来构建神经网络
构建一个神经网络需要以下几步：
- 定义神经网络的`权重`,搭建网络结构
- 遍历整个数据集进行训练
    -将数据输入神经网络
    - 计算loss
    - 计算网络权重的梯度
    - 更新网络权重
        - weight = weight + learning_rate * gradient


```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):#需要继承这个类
    def __init__(self):
        super(Net, self).__init__()
        #建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        #三个全连接层
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): #注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
net
```




    Net (
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear (400 -> 120)
      (fc2): Linear (120 -> 84)
      (fc3): Linear (84 -> 10)
    )




```python
len(list(net.parameters())) #为什么是10呢？ 因为不仅有weights，还有bias， 10=5*2。
                            #list(net.parameters())返回的learnable variables 是按照创建的顺序来的
                            #list(net.parameters())返回 a list of torch.FloatTensor objects
```




    10




```python
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input) #这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
out              #查看源码之后，果真如此。那么，forward()是必须要声明的了，不然会报错
out.backward(torch.randn(1, 10))
```

## 使用loss criterion 和 optimizer训练网络
`torch.nn`包下有很多loss标准。同时`torch.optimizer`帮助完成更新权重的工作。这样就不需要手动更新参数了


```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)  # 有了optimizer就不用写这些了
```


```python
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers，如果不写这个函数，也是可以正常工作的，不知这个函数的必要性在哪？

output = net(input) # 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构

loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
```

## 整体NN结构


```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):#需要继承这个类
    def __init__(self):
        super(Net, self).__init__()
        #建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        #三个全连接层
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): #注意，2D卷积层的输入data维数是 batchsize*channel*height*width
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# in your training loop:
for i in range(num_iteations):
    optimizer.zero_grad() # zero the gradient buffers，如果不写这个函数，也是可以正常工作的，不知这个函数的必要性在哪？

    output = net(input) # 这里就体现出来动态建图了，你还可以传入其他的参数来改变网络的结构

    loss = criterion(output, target)
    loss.backward()
    optimizer.step() # Does the update
```

# 其它

1. 关于求梯度，只有我们定义的Variable才会被求梯度，由`creator`创造的不会去求梯度
2. 自己定义Variable的时候，记得Variable(Tensor, requires_grad = True),这样才会被求梯度，不然的话，是不会求梯度的
3. 


```python
# numpy to Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a) # 如果a 变的话， b也会跟着变，说明b只是保存了一个地址而已，并没有深拷贝
print(b)# Variable只是保存Tensor的地址，如果Tensor变的话，Variable也会跟着变
```


```python
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)# 这个和 a = np.add(a,1)有什么区别呢？
# a = np.add(a,1) 只是将a中保存的指针指向新计算好的数据上去
# np.add(a, 1, out=a) 改变了a指向的数据
```


```python
# 将Tensor放到Cuda上
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y
```
