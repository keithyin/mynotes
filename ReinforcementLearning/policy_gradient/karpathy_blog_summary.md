# policy gradient


$$
\nabla_\theta J(\theta) = \mathbb E_{\pi_\theta} \Bigl[\nabla_\theta\log \pi_\theta(s,a)Q^{\pi_\theta}(s,a)\Bigl]
$$

## 我们希望  最大化$J(\theta)$ 。应该怎么理解这个式子呢？

* $\pi_\theta$ 是当前的 `policy` $\pi_\theta(s,a) = \mathbb P(A_t=a|S_t=s)$
* 先看 $\nabla_\theta\log \pi_\theta(s,a)$ , 这个是 使 $\pi_\theta(s,a) = \mathbb P(A_t=a|S_t=s)$ 概率上升的方向。
* 到底是想要 $\mathbb P(A_t=a|S_t=s)$ 的 概率上升还是下降，还要看 $Q^{\pi_\theta}(s,a)$ 的脸色，如果 为正，那么好，就提升$\mathbb P(A_t=a|S_t=s)$ 的概率，如果 为 负，那好，就降低 $\mathbb P(A_t=a|S_t=s)$ 的概率。
* 如果想要 `policy gradient` work 起来，首先 `policy gradient` 需要很多正例样本引导其去调节正确 `action` 的概率，否则，看多了 `负例` 样本，就在瞎下降，也不概率升高的 `action` 是不是好 `action`。



## 问题来了，如果得知 $Q^{\pi_\theta}(s,a)$ 的值呢？

**$Q^{\pi_\theta}(s,a)$ 是啥？**

* $Q^{\pi_\theta}(s,a)=\mathbb E_{\pi_\theta}\Bigl[G_t|S_t=t, A_t=a\Bigl]$ 
* $G_t$ 是 `discounted reward`。 
* $Q(s,a)$ 值是和 $\pi$ (policy) 紧密相关的，policy 好，$Q$ 值高，反之。
* 所以 $Q^{\pi_\theta}(s,a)$ 可以指示 $\pi$ (policy) 的好坏



**如果估计 $Q^{\pi_\theta}(s,a)$**

* monte-carlo policy gradient



## monte-carlo policy gradient

* using return $v_t$ as an unbiased sample of $Q^{\pi_\theta}(s,a)$

$$
\Delta \theta = \alpha\nabla_\theta\log\pi_\theta(s_t,a_t)v_t
$$

* function REINFORCEMENT
  * initialize $\theta$ arbitrarily
  * for each episode $\{s_1,a_1,r_2, ..., s_{T-1}, a{T-1},r_T\}$ do
    * for $t=1$ to $T-1$ do
      * $\theta \leftarrow \theta+ \alpha\nabla_\theta\log\pi_\theta(s_t,a_t)v_t$
    * end for
  * end for
* end function

意思就是：

* 用当前的 $\pi_\theta$`policy` ,产生一个(或一个 batch) `episode`
* 计算 return $v_t$
* 给这个 `episode` 中的每个 $s_t, a_t$  打上标签， $v_t$
* 然后计算 policy network 的梯度。  
  * 可一个 episode 更新一次
  * 可一个 batch episode 更新一次
* 更新 policy network 的参数
* 回到第一步

