# Dynamic Programming（动态编程）

## 两个问题

* prediction
  * 给定`MDP`, <$\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma$>和`policy` $\pi$， 或者`MRP`, <$\mathcal{S},\mathcal{P}^\pi,\mathcal{R}^\pi,\gamma$>
  * 求得：`value function` $v_\pi$
* control
  * 给定`MDP`, <$\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma$>
  * 求得：最优值函数$v_\star$ 和 最优`policy` $\pi$ 

可以看出，这是两个不同的任务，一个是已知`policy` 求得 值函数。一个是不知`policy`，求得最优的值函数和最优`policy`。



## Iterative Policy Evaluation

**解决prediction 问题**

* 给定`policy` $\pi$
* 求出$v_\pi$

算法：

* 迭代是使用 `Bellman expectation ` 备份
* $v_1->v_2->...->v_\pi$
* 使用同步的 备份
  * At each iteration `k+1`
  * For all states $s\in\mathcal{S}$
  * Update $v_{k+1}(s)$ from $v_k(s')$
  * where $s'$ is a successor state of $s$
* 可以收敛到$v_\pi$

**对应的数学公式为：**
$$
\begin{aligned}
v_{k+1}(s)&=\sum_{a\in A}\pi(a|s)\Bigr( \mathcal{R}_s^a+\gamma\sum\mathcal{P}_{ss'}^av_k(s')\Bigr)\\
v^{k+1}&=\mathcal{R}^\pi+\gamma\mathcal{P}^\pi v^k
\end{aligned}
$$
