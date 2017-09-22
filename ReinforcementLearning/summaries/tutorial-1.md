# 强化学习

**什么是强化学习：**





**数学模型**

* MP
* MDP
* POMDP

**问题**

* reinforcement learning
  * prediction
  * control
* planning
  * prediction
  * control

**planning 解决方法 model-based（dynamic programming）**

* prediction : （计算当前 policy 的 state-value / action-value）
  * policy evaluation （通过迭代法计算当前的 policy 的 state-value / action-value）
* control : （求出最优的 policy）
  * policy iteration : (先 policy evaluation 再 policy improvement)
  * value iteration：

**reinforcement learning 解决方法， model-free**

* prediction (应该也是有 on-policy / off-policy 之分的)

  * MC （采样 trajectory 来估算 state-value / action-value），通过采样来 backup 
  * TD （通过 bootstrap 来估算 state-value / action-value）

* control

  * Value-function （先 prediction，然后在 policy improvement（epsilon-greedy））

    * MC (on-policy / off-policy)  
    * TD (on-policy / off-policy)
    * sarsa ：TD(0) + on-policy
    * q-learning：TD(0) + off-policy
    * DQN：TD(0) + off-policy (**DRL**)

  * policy gradient

    * stochastic
      * REINFORCE :  MC  (**DRL**)


    * continuous

  * actor-critic : 

    * stochastic:
      * REINFORCE with base line : TD(0)  (**DRL**)
      * ​
    * continuous
      * DDPG : policy-gradient + TD(0)  + off-policy (**DRL**)
      * A3C :  policy-gradient + TD(0) + on-policy   (**DRL**),   `这是个框架，可以把很多算法塞进去`
      * ​



**function approximation**

* 为什么提出



**Deep Reinforcement Learning**

> DL 做 function approximation









## reinforcement learning



