# DPG (deterministic policy gradient algorithms)

* **deterministic policy gradient** algorithms for reinforcement learning with **continuous actions**.



* The deterministic policy gradient has a particularly appealing form:
  * the expected gradient of the action-value function.



* 如何保证 确定性 策略梯度下降 有足够的 **exploration**
  * introduce an off-policy actor-critic algorithm
  * behavior policy 是 随机policy， target policy 是 确定性 policy
  * ​



**Keywords**

* deterministic policy
* explore



**Policy Gradient Theorem**
$$
\nabla_\theta J(\pi_\theta) = \mathbb E_{s\sim\rho^\pi, a\sim\pi_\theta}\Bigr[\nabla_\theta\log\pi_\theta(a|s)Q^\pi(s,a)\Bigr]
$$

* policy gradient 的问题在于 如何 估计  $Q^\pi(s,a)$ 的值。



**compatible function approximator  $Q^w(s,a)$**

* $Q^w(s,a) = \nabla_\theta \log \pi_\theta(a|s)^\top w $
* the parameter $w$ are chosen to minimise the mean-squared error $\varepsilon^2(w)=\mathbb E\Biggr[\Bigr(Q^w(s,a)-Q^\pi(s,a\Bigr)^2\Biggr]$
* 通常情况下，第二条限制会放松一下，使用 TD 算法来 估计 $Q^w(s,a)$



**off-policy**

* behavior policy : generate trajectories (episodes)
* target policy : .... 



## Policy Gradient Framework

**首先是一个 stochastic gradient theorem**
$$
\begin{aligned}
\nabla_\theta J(\pi_\theta) &=\int_\mathcal S\rho^\pi(s)\int_\mathcal A \nabla_\theta\pi_\theta(a|s)Q^\pi(s,a) \\
&= \mathbb E_{s\sim\rho^\pi, a\sim\pi_\theta}\Bigr[\nabla_\theta\log\pi_\theta(a|s)Q^\pi(s,a)\Bigr]
\end{aligned}
$$

* $\rho(s)$  discounted state distribution  or **stationary distribution of an ergodic MDP**



**然后是 如何搞定 $Q^\pi(s,a)$**





## 为什么 policy-gradient

多数 model-free 强化学习算法都是基于 `policy-iteration` 的，即：

- 先进行 `policy-evaluation`
- 然后用，`policy-improvement`  $\pi^{\mu^{k+1}}(s)=\arg\limits_a \max Q^{\mu^k}(s,a)$

现在问题来了，如果 $a$ 是连续的话，还咋找 全局最大值，有点痛苦。这时候就是 policy-gradient 方法大显身手的时候了

* move the policy in the direction of the gradient of $Q$ , 而不是 找全局最大值，这时是 policy-network
* 如何理解 **the gradient of $Q$** 呢？$Q$ 对 $a$ 的梯度？$Q$ 对 a 的梯度的方向当然是 $Q$ 值上升的方向。所以，如果有 policy-network, 梯度相应的修改肯定没错。会导致 $Q$ 值上升。

$$
\theta^{t+1} \leftarrow \theta^t+\alpha\Biggr[\nabla_\theta Q^{\mu^k}\Bigr(s,\mu_\theta(s)\Bigr)\Biggr]
$$

再进一步：
$$
\theta^{t+1} \leftarrow \theta^t+\alpha\Biggr[\nabla_\theta\mu_\theta(s)\nabla_a Q^{\mu^k}\Bigr(s,a\Bigr)\Biggr|_{a=\mu_\theta(s)}\Biggr]
$$




## other

**stochastic VS deterministic**

* stochastic
  * the policy gradient integrates over both state and action spaces.   为什么？？？？
  * 不需要单独考虑 exploration 的问题，因为随机 policy 本身就拥有很牛的 exploration 能力
  * require more samples
* deterministic
  * only integrates over state spaces.  为什么 ？？？？
  * 需要考虑 exploration 问题。
  * require less samples



## summary

* 在高维任务中，确定性 policy 的性能要比 随机 policy 的性能要好

