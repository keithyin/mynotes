# Reinforcement Learning

**什么是强化学习**

> Reinforcement Learning is learning what to do -- **how to map situations to actions** -- so as to maximize a numerical reward signal.
>
> **trial-and-error search and delayed reward **-- are the two important distinguishing features of reinforcement learning.
>
> 强化学习是定义在一类问题上，而不是方法上，所有能解决那类问题的方法都可以称作强化学习方法。
>
> 强化学习定义在什么的问题上呢？  a learning agent interacting with its environment to achieve a goal.



在 DRL 中，深度学习那一块可以看作特征提取工具。

几个重要概念：

* state： The state is sufficient statistic of the environment and thereby comprises all the necessary information for the action to take the best action.


> 对 state 的理解： state 提供足够的信息能够引导我们 做出正确的action ，这就够了。
>
> 因为 observation 不等价与 state，这就涉及到如何将 observation （和 action）编码成 state 的方法了，感觉应这么考虑：
>
> * 当前这个任务，如果想要做出正确的 action，需要哪些信息
> * 通过如何 处理 observation 可以得到所需要的信息。
>
> 举个例子 --> Atari Pong：
>
> * 如果想要正确的控制 挡板，我们应该需要 小球的运动方向和 运动速度 和 位置
> * 单一帧只能获得 小球的位置，运动方向和速度都无法获取，所以用 4 帧来代表状态
> * 因为从四帧中 是可以推断出，运动方向，位置，速度的。










## 强化学习

**三大类算法**

* value based --- 现在基本就是 Q-Learning 了
* policy based --- policy gradient 和 actor-critic
* model based




**算法的基本组件**

* MC
* TD(0)
* TD($\lambda$)



**算法的两种性质**

* on-policy
* off-policy




**两类问题**

* prediction
* control



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


**另一种分类方法**

* DQN
  * MC
  * TD
  * on-policy / off-policy
  * stochastic / deterministic policy
  * discrete / continuous policy
* Policy Gradient
  * MC
  * TD
  * on-policy / off-policy
  * stochastic / deterministic policy
  * discrete / continuous policy





## DRL 面临的问题

* 监督信号只有一个 reward，而且十分稀疏
* agent 的 observation 是时序相关的，并不是 iid 的。
  * 这个问题是这样：传统的 RL 算法，都是看到一个 obs，然后直接就更新参数，但是 DL 需要训练数据是 IID 的。用传统 RL 的训练方法显然是不行的啦，所以搞出了 experience replay 方法。
  * 为什么 DL 需要的训练数据是 IID 的呢？ 可能的原因是：因为我们用 mini-batch 训练方法，一个 mini-batch 的梯度应该是 整个 batch 的无偏估计，数据 IID 的话，是 无偏，但是如果数据不是 IID 的话，那就不是 无偏了。
* 如果不好定义 reward，就基本上歇菜了




## Policy

Policy 有两种属性：

* continuous， discrete
* stochastic，deterministic（对于 deterministic policy 的算法一般要求 policy 是 continuous）





## Value Function

Value Function 有两种： **目的是求 bellman equation**

* state-value Function
* action-value Function




## on/off policy

* off-policy：可以保证 exploration




## on/off-line

* ​




## MC-TD bias Variance：

* 为什么说 MC 方法 0-bias， high variance
* 为什么说 TD(0) 方法 low-bias，low variance



**从值估计的角度来理解？**

假设 trajectory $\tau = \{s_0,a_0,s_1,a_1,...\}$ 
$$
p(\tau) = p(s_0)\prod_{t=0}^T \pi(a_t|s_t)p(s_{t+1}|s_t,a_t)
$$

$$
G(s_t)=\sum_t^T r_{t+1}+\gamma r_{t+2} + ... + \gamma^{T-t-1} r_T
$$

$$
R(s_t) = \mathbb E\Bigr[G(s_t)\Bigr]
$$

最后一个式子是对 trajectory 的期望。

MC 方法是采一个 trajectory，所以是对 Value 的无偏估计。但是为什么方差大呢？因为 trajectory 会跑偏？

为什么 TD(0) 方差小？ TD target  $r_{t+1}+V(s')$  ，$s'$ 的取值也是有一个分布的吧，不过这个似乎比 trajectory 方差要小一点，但是引入了偏差，因为 $r_{t+1} + V(s')$ 并不是 $V(s)$ 的无偏估计，只有 $V(s)=r_{t+1}+V(s')$ 时，才是无偏估计。 



从 trajectory 的角度来理解？



## 减小 variance 的方法

> 减小 variance 一般是对于 MC 方法而言的，因为 MC 方法方差大。

* 用 TD(0), 不用 MC 方法
* reward 乘个系数



**reward 乘个小于 1 的系数**

这个方法的直观解释是，MC 采样，越往后方差越大（一步错，步步错，就是这种感觉），然后对 discounted reward 再进行 discounting。 但是这个方法引入了 偏差。为什么呢？

state-value 的定义是：
$$
V(s_t) = \mathbb E\Biggr[r_{t+1}+\gamma r_{t+2}+...\Bigr|s_t\Biggr]
$$
直接用采样的结果来计算 $V(s_t)$ 的话，是无偏估计，但是 对于reward 再进行一次 discount 的话，估计的就不是无偏估计了 ，所以会有偏差。



## Learning & Planning & Search

* Learning : **model is unknown**, learn value function / policy from the experience
* Planning : **model is known**, learn value function / policy from the model
* Search : select the best action of current state by **lookahead**

**Search:** 另一种 Planning 的方法。

* 不用求解整个 MDP， 只需求解 sub-MDP（from now on）
* ​




## Glossary

* prediction problem : 也叫做 policy evaluation。给定一个 policy， 计算 state-value function 或 action-value function 的值。
* control problem ： 寻找最优 policy
* Planning：根据 model 构建一个 value function 或者 policy。（model已知哦）
* on-policy： evaluate or improve the behavior policy
* off-policy ：从 behavior policy 形成的 traces 中学习 自己的最优 policy
* on-line mode：training algorithms are executed on data acquired in sequence。
* off-line mode：也叫 batch mode，并不是看到一个样本就训练。完成一个 epoch 再更新也叫 off-line
* model-free： agent 直接从 experience 中学习，model未知（不知道 状态转移矩阵）


* episodic：环境有个终止状态（而且一定会到达这个终止状态）
* non-episodic： 环境没有终止状态（或者不会到达一个终止状态）（MC的方法歇菜）
* reparameterization trick : [https://www.quora.com/What-is-the-reparameterization-trick-in-variational-autoencoders](https://www.quora.com/What-is-the-reparameterization-trick-in-variational-autoencoders)
* [path-wise derivative](http://www.mathfinance.cn/pathwise-derivative-vs-finite-difference-for-greeks-computation/)
* ​stationary environment : 
* changing environment :
* episode: 从开始到结束一个的 trajectory
* trajectory：轨迹，任意连续时刻都可以构成 trajectory
* revealed information :  揭露的信息
* side information ：(additional variables that are not predicted, but are related to variables that are predicted)
* ​




## 推公式的 几 个trick

* 对期望求导时，将期望改成 求和形式
* score function
* 将对 trajectory 的操作变成对 state-action 的操作。 






## 参考资料

[David Silver](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)

[Berkely RL CS294](http://rll.berkeley.edu/deeprlcourse/#lecture-videos)

[Pong From Pixels karpathy](http://karpathy.github.io/2016/05/31/rl/)

[https://www.nervanasys.com/deep-reinforcement-learning-with-neon/](https://www.nervanasys.com/deep-reinforcement-learning-with-neon/)

