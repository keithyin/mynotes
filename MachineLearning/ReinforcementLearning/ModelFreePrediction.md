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
* To evaluate state $s$
* Every **time-step** $t$ that state $s$ is visited in an episode,
* Increment counter $N(s)\leftarrow N(s)+1$
* Increment total return $S(s)\leftarrow S(s)+G_t$
* Value is extimated by mean return $V(s)=S(s)/N(s)$
* By law of large number, $V(s)\rightarrow v_\pi(s)$ as $N(s)\rightarrow \infty$

细节问题:

* Increment total return $S(s)\leftarrow S(s)+G_t$, 这个时候,一次episode 只更新一个$s$ 吗?还是一个episode更新所有这个episode经历过的.


### Incremental Mean

* 一个序列的均值也可以这么计算
$$
\begin{aligned}
\mu_k &= \frac{1}{k}\sum_{j=1}^kx_j \\
&= \frac{1}{k}\Bigr(x_k+\sum_{j=1}^{k-1}x_j\Bigr)\\
&=\frac{1}{k}(x_k+(k-1)\mu_{k-1})\\
&=\mu_{k-1}+\frac{1}{k}(x_k-\mu_{k-1})
\end{aligned}
$$

$x_k-\mu_{k-1}$: 可以看作 `error term `, 是真实值和期望值的差距.

### Incremental Monte-Carlo Updates

* Update $V(s)$ incrementally after episode $S_1,A_1,R_2,...,S_T$
* For each state $S_t$ with return $G_t$
  $$
  \begin{aligned}
  N(S_t) &\leftarrow N_{S_t}+1\\
  V(S_t)&\leftarrow V(S_t)+\frac{1}{N(S_t)}\Bigr( G_t-V(S_t)\Bigr)
  \end{aligned}
  $$
* In non-stationary problems, it can be useful to track a running mean, i.e. forget old episodes.
  $$
  V(s_t) \leftarrow V(S_T)+\alpha\Bigr(G_t-V(S_t)\Bigr)
  $$


## Temporal-Difference Learning

* TD methods learn directly from episodes of experience
* TD is model-free: no knowledge of MDP transitions/rewards
* TD learns from **incomplete** episodes, by **bootstrapping**
* **TD updates a guess towards a guess**

### TD 和 Monte-Carlo 对比

* 目标: learn $v_\pi$ online from experience under policy $\pi$
* Incremental every-visit Monte-Carlo
  * Update value $V(S_t) toward **actual** return $G_t$
$$
V(S_t)\leftarrow V(S_t)+\alpha\Bigr( G_t-V(S_t)\Bigr)
$$

* Simplest temporal-difference learning algorithm:TD(0)
  * Update value $V(S_t)$ toward **estimated** return $R_{t+1}+\gamma V(S_{t+1})$
$$
V(S_t)\leftarrow V(S_t)+\alpha\Bigr(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\Bigr)
$$
  * $R_{t+1}+\gamma V(S_{t+1})$ is called **TD target**
  * $R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ is called **TD error**
