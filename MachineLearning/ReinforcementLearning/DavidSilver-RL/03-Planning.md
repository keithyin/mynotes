Planning : model is known, learn value function / policy from the model

model is known 的含义是：

* 已知 $\mathcal R_s^a$
* 已知 $\mathcal P_{ss'}^a$

两个任务：

* prediction：给定 policy，输出 value function $v_\pi(s)$
* control：输出 最优 policy。
