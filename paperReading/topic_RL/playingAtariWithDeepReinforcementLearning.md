# DQN 阅读笔记（Atari）

## Abstract

* Learn control policies directly from high-dimensional sensory input using reinforcement learning


* the model trained with a variant of Q-learning



## Introduction

**从深度学习的视角来看 Reinforcement Learning 的挑战：**

* 没有大量的标注数据集
* 强化学习 必须要从 一个标量的 `reward` 信号中去学习，这个 `reward` 信号通常是：`sparse，noisy，delayed`
* 时延可能是 上千个 `time-step`，相对于深度学习的 输入-输出对应关系，这种上千个时延的 `reward`让人很是气馁
* 深度学习中，假设数据是 `iid` 的 ，但是在 强化学习中，`state` 之间是强相关的。
* 深度学习中，假设数据是有一个 固定的 隐含分布的，但是在强化学习中，随着算法的改变，分布也会跟着改变。



**本文的牛逼之处：**

* 证明了，CNN 可以克服以上困难，并从复杂 强化学习环境的原生视频数据中 学习到了一个成功的 `control policies` 。
* 使用 `experience replay mechanism` 减轻了 **correlated data** 和 **non-stationary** 问题。



## Background

强化学习研究的是一个 `agent` 与环境 $\mathcal E$ 交互的问题，这篇文章中，环境就是 `Atari emulator`。

* 问题是 POMDP
* 考虑 `actions 和 observations` 的序列：$s_t=x_1,a_1,x_2,a_2,...,a_{t-1},x_t$ ，（$x$代表 observation，$a$代表action）。`RL` 从这些序列中学习到 `policy`
* 假设：在有限的时间步过后，序列一定会 终止。
* 使用 $s_t$ 来作为 时间 $t$ 的状态表示。



**几个定义：**

* discounted return : $R_t=\sum_{t'=t}^T\gamma^{t'-t}r_{t'}$ ，$T$ 是终止时间点
* 最优 `action-value` 函数 $Q^*(s,a)$，在 $S=s$ 状态时，取 $A=a$   `action-value` 得到的最大
* 即，学习策略 $\pi$ 。



**Bellman equation:**(intuition)

* 如果已知当前状态 $s$ 的 下个状态 $s'$  的 $Q^*(s',a')$ 已知。那么，当前状态的最优的最优 $Q$ 函数是
  $$
  Q^*(s,a)=\Bbb E_{s'\in S}\Biggr(r+\gamma\max_{a'}Q^*\Bigr(s',a'\Bigr) \Bigr|s,a\Biggr)
  $$




最原始的方法，是使用 `lookup table` 来存储 $Q(s,a)$ 的值，可以想象，如果 $s,a$ 的数量非常多，全部存储到内存几乎是不可能的。所以本论文用 神经网络来估计 $Q^*(s,a)$ 的值。

**使用神经网络的好处是：**

* 可以适用于 大规模的 RL 实践中
* 有更好的泛化能力，更新 一次，对所有的状态都会有影响




**如何训练 Q-Network：**

* 最小化一系列的 `loss function` $L_i(\theta_i)$    $i$ 代表 $iteration$ ，$iteration$ 又代表啥呢？
  $$
  L_i(\theta_i)=\Bbb E{s,a\sim \rho (\bullet) }\Biggr[\Bigr(y_i-Q(s,a;\theta_i)\Bigr)^2\Biggr]
  $$

* 其中 $y_i$ 就是在 监督学习中经常见到的 标签，那么 强化学习的 $y_i$ 如何定义呢？
  $$
  y_i=\Bbb E_{s'\in \mathcal E}\Biggr(r+\gamma\max_{a'}Q\Bigr(s',a';\theta_{i-1}\Bigr) \Bigr|s,a\Biggr)
  $$

* 可以看到，是使用 $\theta_{i-1}$ 来定义的第 $i$ 个 $iteration$ 的 $target$ 

* $\rho(s,a)$ 表示 序列中的 $s,a$ 分布，称之为 $behaviour ~distribution$

* 目标函数的导数为
  $$
  \nabla L_i(\theta_i) =\Bbb E_{s,a\sim \rho (\bullet) ,s'\in \mathcal E}\Biggr[\Biggr(r+\gamma\max_{a'}Q\Bigr(s',a';\theta_{i-1}\Bigr)-Q\Bigr(s,a;\theta_i\Bigr)\Biggr)\nabla_{\theta_i}Q\Bigr(s,a;\theta_i\Bigr)\Biggr]
  $$
  ​


## DQN 的前身

**Model Free Control：**

* 不难看出，`dqn` 解决的是一个 `model-free control` 问题。`model-free control` 问题有以下几种解决方法。
  * Monte-Carlo Policy Iteration
  * Monte-Carlo Control
  * GLIE Monte-Carlo Control
  * Sarsa

