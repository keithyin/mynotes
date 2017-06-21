# Multi-Task Learning 概览

对于 [http://sebastianruder.com/multi-task/](http://sebastianruder.com/multi-task/) 的简要翻译。


## Introduction

在机器学习中，我们一般只关心优化 `particular metric`， 也许是 一个在某`benchmark` 上的得分，还是 `business KPI`。为了做这件事，我们通常情况下是 训练一个网络，后者集成（ensemble）多个网络。之后，我们微调这些模型，直到他们的效果不再提升。虽然这样做我们可能会得到一个可以接受的结果，但是我们同时也忽略了可以让模型更好的信息。具体来说，来自`相关任务训练信号` 的信息。通过在相关的任务中共享 `representation` ， 我们可以使我们的网络在原始的任务中泛化的更好。这种方法叫做 `Multi-Task Learning（MTL）`。

`MTL` 在所有的机器学习应用中都被成功的应用， 自然语言处理，语音识别，计算机视觉和 药品发现。 `MTL` 有很多名字：联合学习（`joint learning`），学习去学习（`learning to learn`），通过辅助任务学习（`learning with auxiliary tasks`）。一般的，只要你发现你优化的`loss`多于一个，你就是在做 `MTL`。

即使你仅仅只优化一个损失函数，你也有机会找到一个 辅助任务 去提高模型在主任务上的表现。
> MTL improves generalization by leveraging the domain-specific information contained in the training signals of related tasks.


## motivation

**Motivation:1**
从生物学角度来讲，我们可以认为 `MTL` 是从人类的学习 获得的启发。为了学习新的任务，我们经常使用 从相关任务上获得的知识。例如，一个小孩学习到如何识别人脸，然后他就会将这种知识用在识别其它物体上。
> 这个例子感觉更像迁移学习呢。

**Motivation:2**
从教师的角度来看： 我们通常首先会学习一些 能够提供我们 必要的技巧的任务，然后再去学习更复杂的任务。
> 这个感觉也像迁移学习

**Motivation:3**
从机器学习的角度出发：我们可以把 `MTL` 看作 `inductive transfer` 的一种形式。`Inductive transfer` 可以通过引入一个  `inductive bias` 来提高模型表现，`inductive bias` 使得模型更喜欢某种假设。具体来说，`inductive bias` 的一个常见形式就是 `l1 regularization`， 它使得模型更偏向于稀疏的结果。在`MTL`中， `inductive bias` 是由 辅助任务引入的，这会使得模型会向对两个任务都好的方向发展。这样会使得最终结果 ** 泛化 **更好。
> 相当于引入了正则项

## Two MTL methods for Deep Learning

