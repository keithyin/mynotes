# DQN



**reinforcement learning presents several challenges from a deep learning perspective**

* **DL**: large amounts of hand labelled training data
* **RL**: learn from scalar reward, (sparse noisy delayed)
* **DL**: iid
* **RL**: non-iid

**Solutions:**

* non-stationary distributions:   experience replay to smooth the training distribution   



**Loss Function**
$$
L_i(\theta_i) = \mathbb E_{s,a\sim\rho(\bullet)}\Biggr[\Bigr(y_i-Q(s,a;\theta_i)\Bigr)^2\Biggr]
$$

* $\rho(s,a)$  成为 behaviour distribution，（behaviour policy）

$$
y_i = \mathbb E_{s'\sim \mathcal E}\Bigr[r+\gamma\max\limits_{a'}Q(s', a'; \theta_{i-1})\Bigr]
$$

* $\max\limits_{a'}Q(s', a'; \theta_{i-1})$ 是 target policy。
* $y_i$ 在 叫做 $TD-target$ ，要拟合的目标。但是拟合的时候我们不好好的拟合 瞎鸡巴用 $\varepsilon-greedy$ 找 action, 所以就叫  off-policy。



## Algorithm

* 初始化 replay memory $\mathcal D$ , 容量为 $\mathcal N$
* 初始化 action-value function $Q$ (随即参数)
* for episode =1, M do
  * 初始化 sequence $s_1 = \{x_1\}$, 然后预处理 序列 $\phi_1=\phi(s_1)$  ..???? 看下面解释
  * for t=1, T do
    * 使用 $\varepsilon-greedy$ 算法 挑 $a_t$ , $a_t=random$ 或者 $a_t=\max\limits_aQ^*\Bigr(\phi(s_t),a;\theta\Bigr)$
    * 在模拟器中执行 $a_t$, 会得到一个 reward($r_t$) 和 一个 图像 ($x_{t+1}$) ,**我是多想将 $r_t$ 换成  $r_{t+1}$ 可是原论文是这么写的，无奈。**
    * $s_{t+1} = s_t,a_t,x_{t+1}$ ，然后处理 $\phi_{t+1}=\phi(s_{t+1})$
    * 保存四元组 ($\phi_t, a_t, r_t, \phi_{t+1}$) ，放在 $\mathcal D$ 中，等待 召回
    * 从 $\mathcal D$ 中取出一个 mini-batch  ($\phi_j, a_j, r_j, \phi_{j+1}$) 用来训练
    * 给$y_i$ 值： 当 $\phi_{j+1}$ 是终止状态时是 $y_j=r_j$ , 如果不是终止状态时 $y_j=r_j+\gamma \max\limits_{a'}Q(\phi_{j+1}, a', \theta)$ 
    * 执行梯度下降算法，目标是 $\Bigr(y_j-Q(\phi_j,a_j;\theta)\Bigr)^2$
  * end for
* end for



**总结一下就是：**

* $\varepsilon-greedy$ 算法 往 $\mathcal D$ 中放数据。
* 放一步，取一个 batch 来更新。
* 对更新后的 $Q$  用 $\varepsilon-greedy$ 再来取数据 往 $\mathcal D$ 中放。



**$\phi$ 函数**

* 将原始图片处理成 84*84 大小的图片，图片灰度化
* 然后将 这个 时刻 和前三个时刻的 图片 concat 起来。
* 一共四帧图片，作为 $Q$ 网络的输入。
* 这样就简单的解决了 $POMDP$ ??? 简单暴力。。。。



## 超参数设置

* batch-size: 32
* RMSprop
* $\varepsilon=1$  模拟退火到 $\varepsilon=.1$
* ​



## Evaluation Metric

1. **total reward the agent collects in an episode**  (averaged reward)
2. **track Q-value**, how to track, 
   1. **collect a fixed set of states** by running a random policy before training starts and track its Q-value

