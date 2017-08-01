# value function 算法总结


$$
G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1}R_T
$$

$$
V_\pi(S_t) = \mathbb E(G_t|S_t=s)
$$



## Prediction

> known policy  $\pi$

### MC Prediction

> MC 系列算法的特点是，先用当前 policy 跑完一个(或多个) episode 之后， 然后用 episode 来进行更新 episode。 即，MC  算法的执行 是需要一整個 episode 的。

**state-value function**
$$
V(S_t) \leftarrow V(S_t) + \alpha\Bigr(G_t-V(S_t)\Bigr)
$$

* $G_t$    return
* 算法的目标是希望，$V(S_t)$ 尽量接近  $G_t$ (即 return)，因为 $V(S_t)$ 本来代表的就是  return 的 均值
* 在 MC 中，$G_t$ 是由 采样的 `episode` 估计的。



**action-value function**
$$
Q(s_t, a_t) \leftarrow Q(s_t,a_t) + \alpha\Bigr(r(s_t,a_t)+\gamma G_{t+1}-Q(s_t,a_t)\Bigr)
$$


### TD Prediction

> TD 系列算法的特点是， 不需要完整的 episode， 因为我会 bootstrap（哈哈哈）。TD 干的事情就是  take one step look ahead yo..

**state-value function**
$$
V(S_t) \leftarrow V(S_t)+\alpha\Bigr(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\Bigr)
$$

* 使用 $R_{t+1}+\gamma V(S_{t+1})$ 来 估计 $G_t$ 
* $R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ 称为  $TD$ error
* using the current policy, take one step look ahead to get the  value of $R_{t+1}$, and next state $S_{t+1}$

**action-value function**
$$
Q(s_t, a_t) \leftarrow Q(s_t,a_t)+\alpha\Bigr(r(s_t,a_t)+\gamma Q(s_{t+1}, a_{t+1})-Q(s_t,a_t)\Bigr)
$$




## Control

> Control 有两种策略：
>
> 先 prediction， 然后用 greedy 算法来 选择最优策略
>
> 直接找最优策略（跳过了 prediction 的步骤，或者说一步 prediction 就可以 control）

### MC Control

1. 先 prediction
2. 用 greedy 的方法 进行 policy update

**GLIE**

### TD Control

**Sarsa**  on-policy

> 由于是 TD，所以不需要有整个 episode 也可以训练

policy ： 从 $Q(s,a)$ 中 使用 $\varepsilon-greedy$ 算法来找 action



算法：

* 初始化 $Q(s,a)$ ， $Q(terminal\_state, \star) = 0$ ，$s\in\mathcal S, a\in\mathcal A$
* for each episode:
  * 初始化 $s$
  * 对 $Q(s,a)$ 使用 ($\varepsilon-greedy$) 算法 , 从 $\mathcal A$  中挑一个 action $a$
  * for each step of episode
    * take action $a$, 观察到 $R$ 和 $S'$, 现在我们就有了 ($s,a,r,s'$)
    * 对 $Q(s',a)$ 使用 ($\varepsilon-greedy$) 算法 , 从 $\mathcal A$  中挑一个 action $a'$ , 现在我们有了 ($s,a,r,s',a'$)
    * $Q(s,a)\leftarrow Q(s,a)+\alpha\Bigr[r+\gamma Q(s',a')-Q(s,a)\Bigr]$, 使用已经存在 的 ($s,a,r,s',a'$) 来进行更新。
    * $s \leftarrow s',  a\leftarrow a'$
  * end episode
* end training



**Q-learning** off-policy



## 总结

* **MC** : 无偏，高方差
* **TD**: 有偏， 低方差

