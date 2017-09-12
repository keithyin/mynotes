# Reinforcement Learning

**什么是强化学习**

> Reinforcement Learning is learning what to do -- **how to map situations to actions** -- so as to maximize a numerical reward signal.
>
> **trial-and-error search and delayed reward **-- are the two important distinguishing features of reinforcement learning.
>
> 强化学习是定义在一类问题上，而不是方法上，所有能解决那类问题的方法都可以称作强化学习方法。
>
> 强化学习定义在什么的问题上呢？  a learning agent interacting with its environment to achieve a goal.



## 强化学习

**三大类算法**

* value based
* policy based
* model based



**三类问题**

* prediction
* control
* planning



**算法派系**

* Q-Learning
  * Q-Learning
  * Deep Q-Learning
  * Double Deep Q-Learning
  * Prioritized Experience Replay
* Policy-Based
  * Actor-Critic
  * REINFORCE
  * A3C
  * DPG
  * DDPG
  * ​





## Glossary

* prediction problem : 也叫做 policy evaluation。给定一个 policy， 计算 state-value function 或 action-value function 的值。
* control problem ： 寻找最优 policy
* Planning：根据 model 构建一个 value function 或者 policy。（model已知哦）
* on-policy： evaluate or improve the behavior policy
* off-policy ：从 behavior policy 形成的 traces 中学习 自己的最优 policy
* model-free： agent 直接从 experience 中学习，model未知（不知道 状态转移矩阵）
* on-line mode：training algorithms are executed on data acquired in sequence。
* off-line mode：也叫 batch mode，并不是看到一个样本就训练。