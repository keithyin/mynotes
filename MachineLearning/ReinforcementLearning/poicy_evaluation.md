# Doubly Robust Policy Evaluation and Learning
> Contextual bandit policy evaluation

在contextual bandit中，有两类方法来解决off-line学习问题：
1. direct method: 
	1. estimate the reward function from given data 
	2. and uses this estimate in place of actual reward to evaluate the policy value on a set of contexts.
2. inverse propensity score:
	1. uses importance weighting to correct for the incorrect proportions of actions in the historic data.
  
direct method: 需要一个准确的 reward 模型。但是因为新的policy取action的分布与 旧policy不一致。所以就会导致，旧数据学习出来的reward对于旧policy可能预估的很好，但是对于新的policy并不准确。

inverse propensity score: 旧policy的 propensity score一般能够很好的学习。但是 inverse propensity score 方法的 方差很高。
doubly robust: dicrect method & inverse propensity score有一个模型是正确的，那么 doubly robust 得到的结果就是无偏的。

## direct method
1. 从 $\pi_0$ 采集的数据集中训练一个 $\hat r(s, a)$ 模型
2. 评估 $\pi_1$, $\hat V_{DM}^{\pi_1} = \frac{1}{|S|}\sum_{x\in S} \hat r(s, \pi_1(s))$
