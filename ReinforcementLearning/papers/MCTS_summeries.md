# MCTS summaries

solve sub-MDP

**key point**

* forward search
* sampling



## UCT

**roll-out based Monte-Carlo planning algorithm**

* builds its lookahead tree by **repeatedly** sampling episodes from the initial state.
* 每个 episode is a sequence of **state-action-reward** triplets that are obtained using the domains generative model.
* The tree is built by adding the information gathered during an episode to it in an incremental manner.
* ​



## Deep Learning for real time Atari Game Play Using off-line Monte-Carlo Tree Search Planning



* Monte-Carlo tree search 提供标签
* 用来训练 Value network
* ​



## 问题

* 每次到达一个 state ，都要执行一遍 Monte-Carlo tree search 吗
* 之前 tree search 的保存的状态对 下一次的 tree search 有影响吗
* ​