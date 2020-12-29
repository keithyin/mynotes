# Model-Free Prediction

什么是Model-Free
* $\mathcal P_{ss'}^a$ 未知，即 状态转移矩阵未知
* $\mathcal R_s^a$未知，即 reward未知

Model Free Prediction: 评估一个policy的好坏。仅仅是评估


# 两大方法
* Monte-Carlo: 啥是 Monte-Carlo，实际就是采样
* TC: 

# Monte-Carlo
* MC methods learn directly from **episodes of experience**
* MC is model-free: no knowledge of MDP transitions / rewards
* MC learns from **complete episodes**: no bootstrapping
* MC uses the simplest possible idea: **value = mean return**!!!
* Caveat: can only apply MC to episodic MDPs. All episodes must terminate

目标：学习 $v_\pi(s)$。
回忆一下Return的计算方法 $G_t=R_{t+1}+\gamma R_{t+2} + ... + \gamma^{T-1}R_T$
回忆一下 value function的计算公式：$v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]$

通过 value function的计算公式，我们可以使用采样的方式来计算 empirical mean return
由于，一个episode中，一个 $state$可能出现多次，所以 Monte-Carlo方法 又可以分为 first-visit & every-visit

算法具体执行流程：
* 使用 policy $\pi$ 与环境进行交互，搞到一堆 episodes
* 遍历每个 episodes 的 每个 time step
* Increment counter $N(s) \leftarrow N(s) + 1$
* Increment total return $S(s) \leftarrow S(s) + G_t$
* Value is estimated by mean return $V(s) = S(s)/N(s)$
* 最终, 当$N(s)\rightarrow \inf$ 时 $V(s) \rightarrow v_\pi(s)$
