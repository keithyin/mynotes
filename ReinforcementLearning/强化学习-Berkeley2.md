# 强化学习-笔记2

## Terminology & Notion

* `o`: observation , `u`: why u?

* 对于一般的监督学习框架，可以描述成：$\pi_\theta(u|o)$ ，$\theta$是模型参数

* 关于序列预测问题，就需要时间了，原始的监督学习框架变为$\pi_\theta(u_t|o_t)$

* 在真实的序列预测问题之中，$u_t$ 可能会影响到你的观测 $o_{t+1}$

* $o_t$ --> `observation`

* $u_t$ --> `action`

* $\pi_\theta(u_t|o_t)$ --> policy， 告诉你，当观测到$o_t$时，应该做出什么样的反应$u_t$

* $x_t$ --> `state`。

* $o_t与x_t$ 之间的关系，$o_t$是我们实实在在看到的，$x_t$是导致$o_t$的东西

  <img src="imgs/reinforcement1.png" width="700px">

* $c(x_t, u_t)$ --> 损失函数 `cost function`

* $r(x_t, u_t)$ --> `reward function` 。`reward function = -cost`

* ​

## sequential decision problems



## imitation learning

* 监督学习中，样本需要`IID`， 才能`work`。
* 如何使   $p_{data}(o_t)=p_{\pi_\theta}(o_t)$ ?   `DAgger`
  * 从 $p_{\pi_\theta}(o_t)$ 采集数据，而不是从 $p_{data}(o_t)$ 中
  * 直接执行 $\pi_\theta(u_t|o_t)$ ，但是需要标记$u_t$
  * ​
* 模仿学习的损失函数
  * $c(x,u) = -\text{log}p(u=\pi^*(x)|x)$ 
  * $\pi^*(x)$   人类  `policy`

## case studies of recent work in (deep) imitation learning

## what is missing from imitation learning



## reinforcement learning



$$min \sum_{t=1}^TE[c(x_t, u_t)] $$

$x_{t} \text{~}p(x_{t}|x_{t-1},u_{t-1})$



强化学习（表示一种方法时）忽略模型 $x_{t+1} \text{~}p(x_{t+1}|x_t,u_t)$ 



# 强化学习-笔记3

* can the machine make its own decisions?
  * How can we choose actions under perfect knowledge of the system dynamics?
  * Optimal control, trajectory optimization, planning

## Making decisions under known dynamics?

* $c(x_t, u_t)$ 
* $min \sum_{t=1}^Tc(x_t, u_t)$
* $x_{t} \text{~}p(x_{t}|x_{t-1},u_{t-1})$  , $x_t = f(x_{t-1}, u_{t-1})$



**术语：**

* shooting method: optimize over actions only
* collocation method: optimize over actions and states, with constraints

## Trajectory optimization: bp through dynamical systems



## Linear dynamics: linear-quadratic regulator(LQR)

* shooting method

## Nonlinear dynamics



## Discrete systems:

## Case study

