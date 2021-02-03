# Doubly Robust Policy Evaluation and Learning
> Contextual bandit policy evaluation

在contextual bandit中，有两类方法来解决off-line学习问题：
1. direct method: 
	1. estimate the reward function from given data 
	2. and uses this estimate in place of actual reward to evaluate the policy value on a set of contexts.
2. inverse propensity score:
	1. uses importance weighting to correct for the incorrect proportions of actions in the historic data.
  
direct method: 需要一个准确的 reward 模型。但是因为新的policy取action的分布与旧policy不一致。所以就会导致，旧数据学习出来的reward对于旧policy可能预估的很好，但是对于新的policy并不准确。

inverse propensity score: 旧policy的 propensity score一般能够很好的学习。但是 inverse propensity score 方法的 方差很高。
doubly robust: dicrect method & inverse propensity score有一个模型是正确的，那么 doubly robust 得到的结果就是无偏的。

## Direct Method
1. 从 $\pi_0$ 采集的数据集中训练一个 $\hat r(s, a)$ 模型
2. 评估 $\pi_1$, $\hat V_{DM}^{\pi_1} = \frac{1}{|S|}\sum_{x\in S} \hat r(s, \pi_1(s))$.
3. 评估得到的 $\hat V_{DM}^{\pi_1}$ 与 $\frac{1}{|S|}\sum_{x\in S} r_{s,a}$ 进行比较，看看新策略是好还是坏

## Inverse Propensity Score
$$
\begin{aligned}
\hat V_{IPS}^{\pi_1} &= \mathbb E_{\pi_1(s,a)}\Bigr[f(s, a)\Bigr] \\\\
&= \sum_{s,a} \pi_1(s,a)f(s,a) \\\\
&= \sum_{s,a} \pi_0(s,a)\frac{\pi_1(s,a)}{\pi_0(s,a)}f(s,a) \\\\
&= \mathbb E_{\pi_0(s,a)}\Bigr[\frac{\pi_1(s,a)}{\pi_0(s,a)}f(s, a)\Bigr]
\end{aligned}
$$
得到最后一个公式后，我们就可以使用 $\pi_0$ 的采样数据进行预估了。

## Doubly Robust Estimator
$$
\hat V_{DR}^{\pi_1} = \frac{1}{|S|} \sum_{(s, a, r_a) \in S} \Bigr[ \frac{(r_a  - \hat r(s, a))\pi_1(s,a)}{\pi_0(s,a)} + \mathbb E_{a' \in \pi_1(s,a')}\hat r(s, a') \Bigr]
$$

该论文的实验可以关注一下。

# Doubly Robust Off-policy Value Evaluation for Reinforcement Learning
> multi step mdp policy evaluation

对于policy evaluation来说：infinite mdp 也可以直接简化为 finite mdp. 取的长度 on the order$O(\frac{1}{1-\gamma})$

## Basic Setting
Data: a set of length-H tragectories is sampled using a fixed stochastic policy $\pi_0$, known as behavior policy
Goal: estimate $v^{\pi_1, H}$, the vaue of a given target policy $\pi_1$ from data trajectories.


两类estimator：regression estimators, importance sampling estimators
## Regression Estimators
> 与 contextual bandit 的 direct method 一致
如果 true parameters of the MDP are known. the value of the target poicy can be computed recursively by the Bellman equaltions.

let $V^0(s) = 0$, and for $h=1,2,3,...,H$

$$
\begin{aligned}
Q^h(s,a) &:= \mathbb E_{s' \sim P(s'|s,a)}[R(s,a) + \gamma V^{h-1}{s'}] \\\\
V^h(s) &:= \mathbb E_{a \sim \pi_1(a|s))}[Q^h(s, a)]
\end{aligned}
$$

算法流程：
1. 使用样本回归出来一个 $\hat p(s'|s, a), \hat r(s, a)$. 
2. 然后使用上面的公式进行进行计算就可以了。

## Importance Sampling Estimators
$$
V_{IS} := \rho_{1:H} * (\sum_{t=1}^H\gamma^{t-1}r_t)
$$

$$
V_{step-IS} := \sum_{t=1}^H \gamma^{t-1} \rho_{1:t} r_t
$$

给定一个数据集 $D$, is estimator的结果为 $\frac{1}{|D|}\sum_{i=1}^{|D|} V_{IS}^(i)$. 其中，$|D|$ 表示 trajectory 的的数量。
