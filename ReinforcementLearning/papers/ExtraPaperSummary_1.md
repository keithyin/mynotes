# RL 不太有亮点的文章总结



## How to discount deep reinforcement learning : Towards new Dynamic Strategies

**考虑了两个点**

* discount factor
* learning rate



**最优策略**

* discount factor 慢慢增加
* learning rate 慢慢减少



**intuition**

> it seems plausible that following immediate desires at a young age is a better way to develop its abilities and that delaying strategies is only advantageous afterwards when pushing longer term goals.

> since instabilities are only severe when the discount factor is high. 
>
> 因为 discount factor高的时候，误差传回的比较大，所以导致更新不稳定，小 learning rate 更加合适。
>
> 为什么 误差传回的大？（难道是因为方差的原因？ trajectory变化比较剧烈？）
>
> 直觉上解释： 只看眼前，可以很快拟合，如果看的远的话，变化就很多，所以更新是不稳定的。



**exploration / exploitation**

* lower discount factor : 可能导致局部最优解
  * 因为 lower discount factor 考虑的更多是 immediate reward
  * 想找 全局最优，必须考虑的 reward 要 long term

**疑问：**

* 最终提出的模型是 on-line 的吗。



## Policy Distillation 

* 老师学习一遍
* 用老师学习结果来教学生



## Dynamic Frame Skip Deep Q Network

* 把 frame skip rate 也为 action ，进行学习
* 之考虑了 4, 20 两个 rate



## 