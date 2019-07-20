# 第二遍读David Silver 强化学习 Slides



**Reward**

* A reward $R_t$ is a **scalar** feedback signal
* 用来指示 agent 在 t 时刻做的怎么样
* agent 的工作是 **最大化 cumulative reward** （注意最大化的是 **cumulative reward**）

> Definition (Reward Hypothesis)
>
> **All** goals can be described by the **maximization of expected cumulative reward**
>
> 但是有些任务的 reward 就比较难以定义



注意 **Environment State** 与 **Agent State** 的概念。

* Environment State:  **environment's private representation**
  * 是环境用来 挑选下一个 observation 和 reward 的
  * Environment State 对于 Agent 来说一般是不可见的
* Agent State: **agent's internal representation**
  * 是 Agent 用来挑选下一个 action 的
  * RL 用的就是 Agent State
  * agent state 可以是 任何 关于 历史（所经历过得事情）的函数 



**RL Agent 的核心组件**

一个 agent 必须包含以下一个或多个组件

* Policy: agent 的行为函数 $\pi(s)$  ，可以是确定性的，也可以是随机的
* Value Function : 用来表示 state-[action] 的 好坏
* Model: **agent** 对于 **环境**的 表示, （在 agent 眼中，环境是啥样的）

这些核心组件也对应了其算法： **Policy-Based, Value-Based, Model-Based**





**Reward, Return, Value Function**

* **reward**: $\mathcal R_s^a = E\Biggr[R_{t+1}\Bigr|S_t=s,A_t=a\Biggr]$ 在当前状态，做了某个action，获得的立即奖励
* **return** : $\mathcal G_t=R_{t+1}+\gamma R_{t+2}+...$ ，某个时刻的 cumulative reward
* **Value Function**: $v_\pi(s)=\mathbb E\Biggr[\mathcal G_t\Bigr|S_t=s\Biggr]$  是 return 的期望 





**为什么会 Optimal Policy 这种概念**

* 环境在这，做不同的操作，获得的 Value 也会不同。举个例子：就像学习，我们两个 action（学习，睡觉），状态一个（教室），reward 就设置为考试高分为1,其余情况为0，在教室中 学习，和在 教室中睡觉，最终获得 return 是不同的，我们的目的就是 寻找最好的 policy，来最大化这个 return。Value 的定义就是 return 的均值。
* ​