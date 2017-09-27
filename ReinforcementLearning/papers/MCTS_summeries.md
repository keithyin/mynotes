# MCTS summaries

solve sub-MDP



**key point**

* forward search
* sampling



## UCT

**roll-out based Monte-Carlo planning algorithm**

* builds its lookahead tree by **repeatedly** sampling episodes from the initial state.
* 每个 episode is a sequence of **state-action-reward** triplets that are obtained using the domains generative model.
  * domains generative model ??????
* The tree is built by adding the information gathered during an episode to it **in an incremental manner**.
* ​

**过程基本上是：采到底，然后给经历过的 state-action 赋值**



**为什么用roll-out based algorithms**

> allow us to **keep track of estimates of the action values** at the sampled states encountered in earlier episodes.






## Deep Learning for real time Atari Game Play Using off-line Monte-Carlo Tree Search Planning



* Monte-Carlo tree search 提供标签
* 用来训练 Value network
* ​



## 问题

* 每次到达一个 state ，都要执行一遍 Monte-Carlo tree search 吗
* 之前 tree search 的保存的状态对 下一次的 tree search 有影响吗




## 思考

MCTS 也是可以学到 Q-value 值的（通过 不断的 simulation），这里的 Q-value 也是可以用 神经网络来 approximate 的吧。



假设现在处于一个 state 上 $s_t$ ，要基于  $s_t$ 搞一次 MCTS，MCTS 后得到一个 action， agent take 这个 action，然后到达了另一个 state （$s_{t+1}$ ） ，基于 $s_{t+1}$ 又要做一次  MCTS。

* 这两次 MCTS 有信息共享吗？
* 需要信息共享一下吗？

