# DPG



**Key Point**

* Policy gradient algorithms typically proceed by **sampling this stochastic policy** and **adjusting the policy parameters** in the **direction of greater cumulative reward**.
* Deterministic policy gradient simple follows the gradient of the action-value function.
  * 确定性 policy 的 gradient 跟着 action-value 的gradient。



> In the stochastic case, the policy gradient integrates over both state and action spaces, whereas in the deterministic case it only integrates over the state space.

为什么这么说呢？ 因为确定性 policy， policy function 输出的是唯一 action，而 stochastic 输出的 policy 是随机的（既然随机，所有的 action-state pair 就都要考虑了。）





> To ensure that our deterministic policy gradient algorithms continue to explore satisfactorily, we introduce an off-policy learning algorithm.

毕竟是 deterministic 的算法，所以 exploration 的能力肯定是不行的，如果没有东西及时拉住 梯度更新方向的话，因为它很有可能往局部最优解走。从这句话可以看出：**off-policy 经常是用来解决 exploration 问题的一个方法。**

exploration ，就是让 agent 什么都经历一下，他才知道什么是最好的。一个整天只吃 馒头的人怎么知道 面包才能合他胃口呢。



**deterministic policy 的直观解释**

> 1. The majority of model-free reinforcement learning algorithms are based on generalized policy iteration:  i.e.   policy evaluation ---- policy improvement
> 2. Policy Evaluation ： 估计 当前 policy 下， action-value 的值（MC，TD 方法都可以）
> 3. Policy Improvement：通过 **估计的 action-value function** 来更新 policy $\pi^{k+1}(s)=\arg \max\limits_a Q^{\pi^k}(s,a)$ 
> 4. 可以看出， 这个 argmax  操作对于 连续的 action 来说还是挺恼火的。
> 5. 将 argmax 操作变成： move the policy in the **direction of the gradient of $Q$ **
> 6. the policy parameters $\theta^{k+1}$  are updated in proportion to the gradient $\nabla_\theta Q^{\pi^k}(s,\pi_\theta(s))$
> 7. 每个 state 都建议了不同的 policy improvement 的方向。

最终的更新公式为 
$$
\theta^{k+1} = \theta^k+\alpha \mathbb E_{s \sim \rho^{u^k}}\Biggr[\nabla_\theta\pi_\theta(s)\nabla_aQ^{\pi^k}(s,a)\Bigr|_{a=\pi_\theta(s)}\Biggr]
$$
这里有个头疼的期望，但是是可以估计的哟。就跟深度学习里面经常见到的公式一样（唬人的哦）。

这里可以看出，无论是 原始的 policy-iteration 方法，还是 dpg， 都是为了 更新 policy，使得  action-value 的值增大。这两种方法是多么的一致。

