# 算法总结

## 解决序列决策问题的方法

**通过 value-function**

* 先得到最优  value-function，再用 greedy 算法找 action
* 如何 找到最优 value-function
  * Q-learning， DQN
  * TD
  * MC

**通过 policy **





## Standard Q Learning

**参数化的 value-function**
$$
Q(s,a;\theta_t)
$$
**如何更新参数：**

* in state $S_t$  ，选择一个 action $A_t$ ，观察到 immediate reward $R_{t+1}$  和 下一个 state $S_{t+1}$

$$
\theta_{t+1} = \theta_t+\alpha\Bigr(Y^Q_t-Q(S_t,A_t;\theta_t)\Bigr)\nabla_{\theta_t}Q(S_t, A_t;\theta_t)
$$

**其中：**
$$
Y^Q_t \equiv R_{t+1}+\gamma \max_aQ(S_{t+1}, a; \theta_t)
$$
**目的：**

将 $Q(S_t,A_t;\theta_t)$ 的  值往 $Y^Q_t$ 上靠



## Deep Q-Learning

**将深度学习引入 Reinforcement Learning 存在的问题** 

* 不稳定，原因是
  * 序列样本之间的 相关性
  * 改变 $Q$ 会改变 数据样本 分布( take 不同的 action，当然会看到不同的 sample)
  * action-value  $Q$  和  target-value $r+\gamma\max\limits_{a'}Q(s', a')$ 之间的相关性（意思是，action-value 变，target-value 也会跟着变，变起来没完没了了。）

**DQN 是如何解决以上问题的**

* 序列样本之间的相关性： experience replay， 同样也平滑了数据的分布
* 使用 iterative update 将 $Q$  向 target-value 调整。target-value 只是周期性的调整。这种策略减小了 action-value 和 target-value 的相关性。



**DQN 在第 `i` iteration 的loss function为**
$$
L_i(\theta_i) = \mathbb E_{(s,a,r,s')\sim U(D)}\Biggr[\Biggr(r+\gamma\max_{a'}Q(s',a';\theta^-_i)-Q(s,a;\theta_i)\Biggr)\Biggr]
$$

* $\theta_i$ 是 第 i 步，Q-network 的参数，即 value-network 的参数 
* $\theta^-_i$ 是第 i 步，target-network 的参数
* 其实两个网络结构一样，只不过参数不一样而已
* target-network 的参数怎么更新呢？ 每 C 步，将 Q-network 的参数复制过来就 OK
* 2013 版的 DQN，等价于每 1 步，就把 Q-network 的参数复制过来



* 参数话 的 value-function 换成神经网络


* 加入了 experience replay
* 有了 target-network 这个名词

## Double Q-Learning

**conclusion**

* Q-learning algorithm is known to **overestimate action values** under certain conditions
* DQN 也有上面这个问题
* **Double Q-learning** 解决了上面的问题，怎么解决的呢？
* 如何判断 target overestimate 了呢？



**关于 overestimate action values**

* 什么是 overestimate action values ？？？？？？？？
  * target 会 overestimate，就是 $Y_t$


* Thrun 称：overestimate 是由 不够灵活的 function-approximation 和 noise 造成的。
* 本文称：overestimate 是由 action-value 不准确时造成的。和 source of approximation error 无关
* overestimate 可能会影响到学习到的 policy 的性能。



**Q-learning 的问题**

* 使用同样的 value-function 来 select 和 evaluate   action, 会导致选择 overestimated 的 values
* 为什么这样会导致 overestimate value 呢？？？？？？？？？？？？？

**Double Q-Learning VS Deep Q-Learning**

> Double Q-Learning 改变了 Deep Q-Learning 中 $Y_t$ 的计算方法

Deep Q-Learning:    $Q$ 同时用来选择 action 和 evaluate value 值
$$
Y^Q_t \equiv R_{t+1}+\gamma \max_aQ(S_{t+1}, a; \theta_t)
$$
等价于
$$
Y^Q_t \equiv R_{t+1}+\gamma Q(S_{t+1}, \arg\limits_a\max Q(S_{t+1},a;\theta_t); \theta_t)
$$
Double Q-Learning: 一个 $Q$ 用来 选择 action，另一个 $Q$ 用来 evaluate value 值
$$
Y^{DoubleQ}_t \equiv R_{t+1}+\gamma Q(S_{t+1}, \arg\limits_a\max Q(S_{t+1},a;\theta_t); \theta'_t)
$$
**如何判断 Deep Q-Learning overestimate 了**

* 通过学习好的 policy，做出一个 discounted value
* 然后判断在 DQN 在学习过程中，estimate 的 value 和这个学习好的 discounted value 的关系

