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

现在来看一下，两种经常使用的 `MTL` 结构。

* hard parameter sharing
* soft parameter sharing

<center>![img](imgs/mtl_images-001-2.png)</center>
<center>图一：hard parameter sharing</center>

`hard parameter sharing` 是最经常使用的 `MTL` 结构。通过保持几个 `task-specific` 的输出层，然后不同的任务之间共享隐层。`hard parameter sharing` 大大的降低了过拟合的风险。 事实上，[这篇文章](http://link.springer.com/article/10.1023/A:1007327622663) 表明，共享参数 过拟合的风险是 N 次 小于 `task-specific` 参数的（即输出层参数）。这个很容易理解，我们同时学习的任务越多，模型就越趋向于寻找对所有`task` 都友好的表示，就越不可能对某一任务过拟合。

<center>![img](imgs/mtl_images-002-1.png)</center>
<center>图二：soft parameter sharing</center>

`MTL` 的 `soft soft parameter sharing` 方法是由其它模型的  `regularization techniques ` 启发得到的。一会就会讨论到。

## Why does MTL work?

通过 `MTL`可以得到`inducitve bias` 直觉上来说貌似很有道理，但是为了更好的理解 `MTL`， 我们需要看一下底层的机制。对于所有的例子， 我们将假设有两个相关的 任务 $A$ 和 $B$ , 他们依赖于共同的隐层表示 $F$。

### Implicit data augmentation

`MTL` 有效的 增加了训练模型的样本数量。 由于所有的 任务 或多或少会有些噪声， 当我们为任务 $A$ 训练一个模型的时候，我们的目标为任务 $A$ 是学习到一个好的 表示，这个表示可以忽略数据相关的噪声，而且泛化的很好。由于不同 任务具有不同模式的噪声，一个同时学习两个任务的模型 就能够学习到更加通用的表示， 因为同时学习 $A$ 和 $B$ 可以使模型通过平均 `noise patterns` 获得更好的 $F$ 表示。
 > 原文中是 说 As all tasks are at least somewhat noisy。 不知这个 task noisy 理解成 task 数据中的noise 是不是正确的。
 
 
### Attention focusing

如果一个任务非常 `nosy`， 或者数据有限，数据维度很高。 让模型区分哪些特征是相关的，哪些特征是无关的是非常困难的一件事。`MTL` 可以帮助模型专注于那些对其他任务也重要的特征，这样会对哪些特征是有关的，哪些特征是无关的提供额外的证据。

### Eavesdropping （偷听。。？？？）

有些特征 $G$ 对于 $B$ 任务来说是非常容易学习的，但是对于$A$任务来说就难学习了。这可能是因为 $A$ 与特征的交互更加复杂，也可能是因为其它特征妨碍了模型学习 $G$ 特征。通过 `MTL`，我们可以允许模型去偷听，即：通过任务$B$学习特征$G$。 最简单的方法就是直接训练模型去输出最重要的特征。

### Representation bias

`MTL` 倾向于学习到一种多种任务都喜欢的 特征。这可以帮助模型很好的泛化到新任务上。

### Regularization
最后， `MTL` 可以看作为一个正则项，因为它引入了 `inductive bias`。正因如此，它降低了过拟合的风险 和 模型的 `Rademacher complexity`，即：拟合随即噪声的能力。

## MTL in non-neural models

为了更好的理解`MTL`在深度学习中的作用，我们先看一下 `MTL` 在 线性模型，核方法，贝叶斯算法中的应用。特别的，我们将讨论在 `MTL`历史上普遍使用的两个想法： 

* enforcing sparsity across tasks through norm regularization
* modelling the relationships between tasks.

注意文献中的很多`MTL`方法都在处理 `homogenous setting`：他们假设所有的任务都与单一输出相关联，即：多类别的 `MNIST 数据集` 可以转化成10个二进制分类任务。最近的方法处理一些更加实际的 `homogenouts setting`， 即：每个任务和一个输出集合相关联。

### Block-sparse regularization
为了能把下面的方法更好的联合起来，我们首先介绍一个符号：我们有 $T$ 个任务，对于每一个任务 $t$, 我们有一个模型 $m_t$ ，参数为 $a_t$，参数的维度为$d$。我们可以把参数写成一个列向量，  $a_t=$
 
 