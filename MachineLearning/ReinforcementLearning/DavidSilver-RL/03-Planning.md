Planning : model is known, learn value function / policy from the model

model is known 的含义是：

* 已知 $\mathcal R_s^a$
* 已知 $\mathcal P_{ss'}^a$

两个任务：

* prediction：给定 policy，输出 value function $v_\pi(s)$
* control：输出 最优 policy。

# Policy Evaluation

问题： 给定 policy，计算其 value function $v_\pi(s)$
解决方案：iterative application of Bellman expectation backup

算法：
1. at each iteration k+1
2. 对所有的状态
3. 根据 $v_k(s')$ 更新 $v_{k+1}(s)$, $s'$ 是 $s$ 的下一个状态

更新公式

$$
v_{k+1}(s) = \sum_{a \in A}\pi(a|s) \Bigr(  \mathcal R_s^a + \gamma\sum_{s' \in S} \mathcal P_{ss'}^{a'}v_k(s') \Bigr)
$$
 
或者更新公式
$$
q_{k+1}(s,a)=\mathcal R_s^a+\gamma\sum_{s'\in S}\mathcal P_{ss'}^a\sum_{a'\in A}\pi(a'|s')q_{k}(s',a'))
$$

# Policy Iteration

1. at each iteration j+1
1. 进行 policy evaluation（计算 $v_{\pi_j}(s)$）（内部需要迭代 $K$ 次 才能得到正确的 policy evaluation)
2. policy evaluation之后，$v_{\pi_j}(s)$就都知道了。我们可以使用以下公式 improve policy。对所有的状态 使用 $v_{\pi_j}(s)$ 更新 $v_{\pi_{j+1}}(s)$

$$
\pi_{j+1}(a|s) = \max_a \mathcal R_s^a + \gamma\sum_{s'\in S}\mathcal P_{ss'}^a v_{\pi_{j}}(s')
$$
 
或者使用以下更新公式
$$
\pi_{j+1}(a|s) = \max_a \pi_j(a|s)
$$

修改点：
1. policy evaluation 是否真的需要收敛才可以？执行n次 policy evaluation迭代是否可行


# Value Iteration
> idea：如果我们知道子问题的解 $v_{optimal}(s')$，那么原始问题的解通过 one-step look ahead 也可以找到.
> value iteration 就是基于下面这个公式
> 直觉：从最终的 reward 的开始，往前进行迭代

$$
v_{optimal}(s) = \max_a \mathcal R_s^a + \gamma \sum_{s' \in S} \mathcal P_{ss'}^a v_{optimal}(s')
$$

算法：
1. at each iteration k+1
2. 对于所有的状态 $s\in S$
3. 根据上面的公式，通过 $v_{k}(s')$ 更新 $v_{k+1}(s)$

与 policy iteration 的对比：
1. 没有显式的 policy
2. 中间过程的 value functions 可能并不对应任何 policy

# Asynchronous Dynamic Programming
前面聊的几个方法都是 Synchronous backups, 可以看出：我们都是通过 $v_k(s')$ 来更新 $v_{k+1}(s)$ 的。
那么什么是 asynchronous backups 如何操作呢？我们直接 inplace 的更新 $v(s)$.

Three simple ideas for asynchronous dynamic programming:
* In-place dynamic programming (inplace 更新 $v(s)$)
* Prioritised sweeping (因为状态-值函数的更新 和 状态选取的顺序是有关系的，所以如何选取状态是一个值得思考🤔的点)
* Real-time dynamic programming

**Prioritised Sweeping**
* 使用 bellman error 指导状态的选择，error 大的状态优先更新。
$$
\Bigr | \max_a \Bigr (\mathcal R_s^a + \gamma\sum_{s'}\mathcal P_{ss'}^av(s') -v(s) \Bigr )\Bigr |
$$