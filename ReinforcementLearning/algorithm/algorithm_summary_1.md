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
Y^Q_t \equiv R_{t+1}+\gamma \max_aQ(S_{t+1}, A_t; \theta_t)
$$
**目的：**

将 $Q(S_t,A_t;\theta_t)$ 的  值往 $Y^Q_t$ 上靠



## Deep Q-Learning

* 参数话 的 value-function 换成神经网络


* 加入了 experience replay
* 有了 target-network 这个名词

## Double Q-Learning

**conclusion**

* Q-learning algorithm is known to **overestimate action values** under certain conditions
* DQN 也有上面这个问题
* **Double Q-learning** 解决了上面的问题，怎么解决的呢？



**关于 overestimate action values**

* 什么是 overestimate action values ？？？？？？？？


* Thrun 称：overestimate 是由 不够灵活的 function-approximation 和 noise 造成的。
* 本文称：overestimate 是由 action-value 不准确时造成的。和 source of approximation error 无关



**Q-learning 的问题**

* 使用同样的 value 来 select 和 evaluate   action