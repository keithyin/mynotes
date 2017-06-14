# ModelFreePrediction

**目标：** Estimate the value function of an **unknown MDP**

## Monte-Carlo Reinforcement learning
* MC methods learn directly from episodes of experences(从现实实验中学习)
* MC is Model-Free. 意思是, 不需要知道`MDP` 的状态转移矩阵 $\mathcal{P}$ 和奖励函数 $\mathcal{R}$
* MC learns from *complete episodes*: no bootstrapping
* MC uses the simplest possible idea: `value = mean return`
* 附加说明:can only apply MC to *episodic* MDPs
  * ALL episodes must terminate

### Monte-Carlo Policy Evaluation

* Goal: learn $v_\pi$ from episodes of experience under policy $\pi$
  $S_1,A_1,R_2,...,S_k,A_k,R_{k+1}\sim \pi$
* Recall that the *return* is the total discounted reward
  $G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...+\gamma^{T-1}R_T$   $T$ 代表`Terminal`的意思
* Recall that the value function is the expected returns:
  $v_\pi(s)=\Bbb{E}[G_t|S_t=s]$
* Monte-Carlo policy evaluation uses *emperical mean* return instead of *expected return*

**算法一览:First-Visit Monte-Carlo Policy Evaluation**
* To evaluate state $s$
* The **first time-step** $t$ that state $s$ is visited in an episode,
* Increment counter $N(s)\leftarrow N(s)+1$
* Increment total return $S(s)\leftarrow S(s)+G_t$
* Value is extimated by mean return $V(s)=S(s)/N(s)$
* By law of large number, $V(s)\rightarrow v_\pi(s)$ as $N(s)\rightarrow \infty$

细节问题:
* $G_t$ 从何而来
* $N(s)$ 到最后是所有的 `episodes` 吗?


**算法一览:Every-Visit Monte-Carlo Policy Evaluation**
* 
