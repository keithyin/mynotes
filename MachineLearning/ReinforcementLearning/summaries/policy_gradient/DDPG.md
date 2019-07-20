# DDPG 算法



## DQN

**缺点**

* action 不能是连续的
* 即使是使用量化的方法， action 也是很多



**疑问**

* 这个算法中也用到了  dqn， 这 TM。。。。怎么就没有 action 连续的问题了。如何用的呢？




**文中对 原始 DQN 进行了一些修改**

* soft target updates
  * 创建了一个 `actor-critic` 网络的副本（target network），$Q'(s,a|\theta^{Q'})$ 和 $\mu'(s|\theta^{\mu'})$ , 用他们来计算 target value.  这个 target 就是 target policy 的。