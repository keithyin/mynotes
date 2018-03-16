# ignite 

最近自己想写一个高级一点的抽象来更方便的训练 `pytorch` 网络, 但是写来写去总感觉用起来比较别扭, 可能是因为自己代码量不够, 搞不出来比较好的抽象吧. 无意间发现, `pytorch` 用户下面有个 `ignite repo`, 好奇就看了一下这是个什么东西. 原来是 `pytorch` 已经提供了一个高级抽象库来训练 `pytorch`模型了, 既然有了轮子, 那就没必要自己造了, 好好用着就行了. 没事读读源码, 也可以学习一下大佬们是怎么抽象的. 废话不多说, 由于 `ignite` 目前缺少官方文档(`API documentation and an overview of the library coming soon.` ), 所以本博文主要是对 `ignite` 做一个宏观上的介绍.



虽然没有官方文档, 但是官方有 [例子](https://github.com/pytorch/ignite/tree/master/examples), 我们可以通过例子来看一下到底该怎么用 `ignite`



## 例子

为了减少源码篇幅, 特地将与 `ignite` 关系不大的代码给删除了,  如果想跑完整示例的话, 可以查看上面提到的链接.

```python
from argparse import ArgumentParser
from torch import nn
from torch.optim import SGD
from torchvision.transforms import Compose, ToTensor, Normalize

from ignite.engines import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss

def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    cuda = torch.cuda.is_available()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)

    model = Net()
    if cuda:
        model = model.cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': CategoricalAccuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            cuda=cuda)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer, state):
        iter = (state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(state.epoch, iter, len(train_loader), state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer, state):
        metrics = evaluator.run(val_loader).metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(state.epoch, avg_accuracy, avg_nll))

    trainer.run(train_loader, max_epochs=epochs)
```



**先对流程做一下总结, 再看API做了些什么**

* 创建模型, 创建 `Dataloader`
* 创建 `trainer`
* 创建 `evaluator`
* 为一些事件注册函数, `@trainer.on()`
* `.run()`



**create_supervised_trainer**

```python
def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Trainer: a trainer instance with supervised update function
    """
```



**create_supervised_evaluator**

```python
def create_supervised_evaluator(model, metrics={}, cuda=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Evaluator: a evaluator instance with supervised inference function
    """
```



**Trainer**

```python
""" 为某事件注册函数, 当事件发生时, 此函数就会被调用
函数的 signature 必须是 def func(trainer, state)
"""
@trainer.on(...)
def some_func(trainer, state):
    pass

Trainer.run() # 训练模型
```



**Event**

```python
"""
类似枚举类, 定义了几个事件
"""
class Events(Enum):
    EPOCH_STARTED = "epoch_started"               # 当一个新的 epoch 开始时会触发此事件
    EPOCH_COMPLETED = "epoch_completed"           # 当一个 epoch 结束时, 会触发此事件
    STARTED = "started"                           # 开始训练模型是, 会触发此事件
    COMPLETED = "completed"                       # 当训练结束时, 会触发此事件
    ITERATION_STARTED = "iteration_started"       # 当一个 iteration 开始时, 会触发此事件
    ITERATION_COMPLETED = "iteration_completed"   # 当一个 iteration 结束时, 会触发此事件
    EXCEPTION_RAISED = "exception_raised"         # 当有异常发生时, 会触发此事件
```



**State**

```python
class State(object):
    def __init__(self, **kwargs):
        self.iteration = 0            # 记录 iteration
        self.output = None            # 当前 iteration 的 输出. 对于 Supervised Trainer 来说, 是 loss.
        self.batch = None             # 本次 iteration 的 mini-batch 样本
        for k, v in kwargs.items():   # 其它一些希望 State 记录下来的 状态
            setattr(self, k, v)
```



**Evaluator**

```python
# 为 evaluator 一些事件注册 函数.
@evaluator.on(...) 
def func(evaluator, state):
    pass

Evaluator.run() # 执行计算.
Evaluator.metrics # 验证集上 metrics 计算的结果都保存在这里
```



