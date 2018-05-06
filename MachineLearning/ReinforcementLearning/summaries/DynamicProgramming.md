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

## 如何提高`policy` , `control`?

**算法1：**（先进行值函数评估，然后再贪婪的找`policy`）(`Policy Iteration`)， 为什么叫`policy Iteration`

* 给定一个 `policy` $\pi$

  * evaluate the policy $\pi$

    $v_\pi(s)=\Bbb{E}[R_{t+1}+\gamma R_{t+2}+...|S_t=s]$

  * improve the policy by acting greedily with respect to $v_\pi$

    $\pi'=greedy(v_\pi)$

* This process of policy iteration always converges to $\pi^\star$

**算法2:**(`Policy Improvement`)

* Consider a *deterministic* policy, $a=\pi(s)$

* We can improve the policy by acting greedily

  $\pi'(s)=\text{arg} \max\limits_{a\in A}q_\pi(s,a)$ 

* This improves the value from any state $s$ over one step

  $q_\pi(s,\pi'(s))=\max\limits_{a\in A}q_\pi(s,a)\ge q_\pi(s,\pi(s))=v_\pi(s)$

* It therefor improves the value function, $v_\pi'(s)\ge v_\pi(s)$




## 修改一下 `policy iteration` 算法

* `policy iteration`算法中，值迭代的时候必须要 收敛到 $v_\pi$ 吗？

* 或者，我们只需要引入一个提前停止条件

  $\epsilon$-convergence of value function

* 或者，简单的的迭代 $k$ 次就行了

  例如：在`Small grid world` 中，$k=3$ 就足够了

* 或者，一次迭代就 更新 `policy`

  这个等价于 `value iteration`



## 确定性值迭代算法（寻找最优`policy`, control 问题）

* The solution $v_\star(s)$ can be found by one-step look ahead

  $v_\star(s)\leftarrow \max\limits_{a\in A}\Bigr(\mathcal{R}_s^a+\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^av_\star(s')\Bigr)$

* The idea of `value iteration ` is to apply these updates iteratively

* Intuition: start with final reward and work backwards

* Still works with loopy, stochastic MDPs



**Value Iteration 算法：**

* 问题：找到 最优 `policy` $\pi$

* 解决方案：iteration application of **Bellman optimality** backup

* $v_1\rightarrow v_2\rightarrow v_3 \rightarrow ... \rightarrow v_\pi$

* 使用同步的 backups

  * At each iteration k+1
  * For all states $s \in S$
  * update $v_{k+1}(s)$ from $v_k(s')$

* 可以收敛到 $v_\star$

* 不同于`policy iteration` ，这个方法没有显式的`policy` (看到的只有 $v$)

* Intermediate value function may not correspond to any policy

* 更新公式就像下面所示
  $$
  \begin{aligned}
  v_{k+1}&=\max_{a\in A}\Bigr(\mathcal{R}_s^a+\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^av_k(s') \Bigr)\\
  \mathbf{v}_{k+1}&=\max_{a\in A}\Bigr( \mathcal{R}^a+\gamma\mathcal{P}^a\mathbf{v}_k\Bigr)
  \end{aligned}
  $$






## 异步动态编程

* 以上所说的方法都是同步的方法，需要保存整个$\mathbf{v}_k$
* 然后再更新所有 $v_{k+1}(s)  ~~~~~~~~\forall s\in S$
* Asynchronous DP backs up states individually, in any order
* For each selected state, apply the appropriate backup
* Can significantly reduce computation
* Guaranteed to converge if all states continue to be selected




**Three simple ideas for asynchronous dynamic programming:**

* In-place dynamic programming
* Prioritised sweeping
* Real-time dynamic programming




**In-place Dynamic Programming:**

* 同步的方法保存两个`value function` 的拷贝

  * `for all s in S`
    $$
    v_{new}(s) \leftarrow \max_{a\in A}\Bigr(  \mathcal{R}_s^a +\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^av_{old}(s')\Bigr)
    $$

  * $v_{old} \leftarrow v_{new}$

* 异步的方法

  * `for all s in S`
    $$
    v(s) \leftarrow \max_{a\in A}\Bigr(  \mathcal{R}_s^a +\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^av(s')\Bigr)
    $$





**Prioritised Sweeping:**

* Using magnitude of Bellman error to guide state selection
  $$
  \Biggr|\max_{a\in A}\Bigr(  \mathcal{R}_s^a +\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^av(s')\Bigr) -v(s)\Biggr|
  $$

* One step look ahead 之后，看看和当前的`value function`相差多少

* 通过这种方式来挑选下一步的 `state`



**Real-Time  Dynamic Programming**

* Idea: only states that are relevant to agent

* Using agent's experience to guide the selection of states

* After each time-step ($S_t,A_t,R_{t+1}$)

* Backup the state $S_t$
  $$
  v(S_t)=\max_{a\in A}\Bigr(  \mathcal{R}_{S_t}^a +\gamma\sum_{s'\in S}\mathcal{P}_{S_{t}s'}^av(s')\Bigr)
  $$




**Full-Width Backups:**

* DP uses *full-width* backups
* For each backup (sync or async)
  * **Every** successor state and action is considered
  * Using knowledge of the MDP *transitions* and *reward* function
* DP is effective for medium-sized problems (millions of states)
* For large problems DP suffers *Bellman's curse of dimensionality*
  * Number of states n=|$\mathcal{S}$| grows exponentially with number of state variables.
* Even one backup can be too expensive



## Samples Backups

* Using sample *rewards* and sample *transitions* ($S,A,R,S'$)
* Instead of reward function $\mathcal{R}$ and transition dynamics $\mathcal{P}$
* Advantages:
  * Model-free: no advance knowledge of MDP required
  * Breaks the curse of dimensionality through sampling
  * Cost of backup is constant, independent of n=|$\mathcal{S}$|

