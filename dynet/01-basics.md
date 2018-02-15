# Dynet 入门

In neural network packages there are generally two modes of operation:

- **Static networks**： 建立一个网络，然后 feed
- **Dynamic networks**：每次前向运算都会建立一个新的网络。

`Dynet` 包含两种 网络模式。



## Package Fundamentals

`Dynet` 的主要部分是 `ComputationGraph`(计算图)，计算图定义了神经网络。计算图由 `expressions`(表达式)组成，表达式与 输入，输出 和 网络的 `Parameters` 相关。`Parameters`是由 网络来进行优化的，所有的 `Parameters` 由 `ParameterCollection` 管理。`trainers`(例如 `SimpleSGDTrainer`) 负责更新 `Parameters` 的值。

在 `Dynet` 中，我们不会直接使用 `ComputationGraph`，它是个**单例对象** , 静静的呆在后端。当 `dynet` 被 `import` ，一个新的 `ComputationGraph` 被创建。我们通过 `renet_cg()` 来重置 `ComputationGraph`。



## Static Networks

`Dynet` 程序的流程总结如下：

1. 创建一个 `ParameterCollection`， 然后像里面填入 `Parameters`
2. 重置计算图，创建表达式用来表示神经网络（network 中也要包含 Parameters 的 Expression）
3. 通过 目标函数 来优化网络。



**来看一个例子：**

考虑一个模型用来解决 `xor` 问题。网络有两个输入，可以是 0 或 1, 一个输出，表示 `input` 的 `xor` 结果。我们用一个有一个隐层的多层感知机来模拟 `xor` 计算。 

另 $x=x_1,x_2$ 表示输入。隐层 8 个单元，输出 1 个单元。隐层的激活函数是 `tanh`。网络的数学表示为：
$$
\sigma(V\tanh(Wx+b))
$$
$W$ : 8×2 矩阵， $V$ : 8×1矩阵， $b$: 8维 向量。

由于希望输出为 0 或 1,所以输出的激活函数使用 `sigmoid`。



代码如下：

```python
# we assume that we have the dynet module in your path.
# OUTDATED: we also assume that LD_LIBRARY_PATH includes a pointer to where libcnn_shared.so is.
import dynet as dy

# 1. 创建一个 ParametetCollection and populates it with parameters.
m = dy.ParameterCollection()
pW = m.add_parameters((8,2))
pV = m.add_parameters((1,8))
pb = m.add_parameters((8))

# 2. new computation graph. not strictly needed here, but good practice.
dy.renew_cg() 

# associate the parameters with cg Expressions
# creates a computation graph and adds the parameters to it, 
# transforming them into Expressions. 
# The need to distinguish model parameters from “expressions” will become clearer later.
W = dy.parameter(pW)
V = dy.parameter(pV)
b = dy.parameter(pb)

# b.value() 计算表达式的值

x = dy.vecInput(2) # an input vector of size 2. Also an expression.
output = dy.logistic(V*(dy.tanh((W*x)+b)))
```



```python
# we want to be able to define a loss, so we need an input expression to work against.
y = dy.scalarInput(0) # this will hold the correct answer
loss = dy.binary_log_loss(output, y)
```



```python
x.set([1,0])
y.set(0)
print(loss.value())

y.set(1)
print(loss.value())

```



## Training

现在，我们想要调整 `Parameters` 来最小化损失函数。

为了达到这个目的，我们需要创建一个 `trainer` 对象。  A trainer is constructed with respect to the parameters of a given model.

```
trainer = SimpleSGDTrainer(m)

```

训练网络的过程如下：

1. 设置 `placeholder` 
2. `loss` 节点执行 `loss.backward()`
3. 使用 `trainer` 更新 `Parameter`

```python
x.set([1,0])
y.set(1)
loss_value = loss.value() # this performs a forward through the network.
print("the loss before step is:",loss_value)

# now do an optimization step
loss.backward()  # compute the gradients
trainer.update()

# see how it affected the loss:
loss_value = loss.value(recalculate=True) # recalculate=True means "don't use precomputed value"
print("the loss after step is:",loss_value)

```

```
the loss before step is: 0.335373580456
the loss after step is: 0.296859383583
```

下面创建一个 `xor` 数据集用来训练 网络：

```python
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in xrange(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()
```

```python
total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print "average loss is:",total_loss / seen_instances
```

验证网络学习的结果：

```python
x.set([0,1])
print "0,1",output.value()

x.set([1,0])
print "1,0",output.value()

x.set([0,0])
print "0,0",output.value()

x.set([1,1])
print "1,1",output.value()

```

```
0,1 0.998090803623
1,0 0.998076915741
0,0 0.00135990511626
1,1 0.00213058013469
```



## To summarize

Here is a complete program:

```python
# define the parameters
m = ParameterCollection()
pW = m.add_parameters((8,2))
pV = m.add_parameters((1,8))
pb = m.add_parameters((8))

# renew the computation graph
renew_cg()

# add the parameters to the graph
W = parameter(pW)
V = parameter(pV)
b = parameter(pb)

# create the network
x = vecInput(2) # an input vector of size 2.
output = logistic(V*(tanh((W*x)+b)))
# define the loss with respect to an output y.
y = scalarInput(0) # this will hold the correct answer
loss = binary_log_loss(output, y)

# create training instances
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in xrange(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()

# train the network
trainer = SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print "average loss is:",total_loss / seen_instances


```

## Dynamic Networks

`Dynamic networks` 和 `Static networks` 非常类似，区别在与，`Dynamic networks` 每次前向计算都会重新建立一个新的前向图。 

We present an example below. While the value of this may not be clear in the `xor` example, the dynamic approach is very convenient for networks for which the structure is not fixed, such as recurrent or recursive networks.

```python
import dynet as dy
# create training instances, as before
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in xrange(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()

# create a network for the xor problem given input and output
def create_xor_network(pW, pV, pb, inputs, expected_answer):
    dy.renew_cg() # new computation graph
    W = dy.parameter(pW) # add parameters to graph as expressions
    V = dy.parameter(pV)
    b = dy.parameter(pb)
    x = dy.vecInput(len(inputs))
    x.set(inputs)
    y = dy.scalarInput(expected_answer)
    output = dy.logistic(V*(dy.tanh((W*x)+b)))
    loss =  dy.binary_log_loss(output, y)
    return loss

m2 = dy.ParameterCollection()
pW = m2.add_parameters((8,2))
pV = m2.add_parameters((1,8))
pb = m2.add_parameters((8))
trainer = dy.SimpleSGDTrainer(m2)

seen_instances = 0
total_loss = 0
for question, answer in zip(questions, answers):
    loss = create_xor_network(pW, pV, pb, question, answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print "average loss is:",total_loss / seen_instances
```