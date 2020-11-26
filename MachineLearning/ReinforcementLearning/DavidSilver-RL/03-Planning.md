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
