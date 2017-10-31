# MIT self-driving car



## self-driving car tasks

* Localization and Mapping
  * Where am i ?
* Scene Understanding
  * Where is everyone else ?
* Movement Planning
  * How do i get from A to B?
* Driver State
  * What's the driver do?



## 当前神经网络的缺陷

* 缺少 推理 （reasoning） 能力
* 需要大量的数据
* 需要监督信号
* 需要手动 选择网络结构
* 需要调节大量的超参数
  * 学习率 learning rate
  * 损失函数 loss function
  * mini-batch 的大小
  * 训练迭代次数
  * 动量，Momentum
  * 优化器（SGD，Adam，。。。）
* 对于强化学习来说，设计一个好的 reward function 也是非常困难的



## 当前深度学习

* 深度学习就是在学习一个 representation
* **学习 一个好的特征表示是非常重要的**



**深度学习的一些应用**

* 目标非类
* 目标检测
* 语义分割
* 图像上色
* OCR
* text to image， image to text
* ...



**为啥激活函数都是 smooth 的**

* smooth 的话， 权值更改一点点，输出也会变换一点点
* 如果不是 smooth 的话，权值更改 一点点，输出可能变化非常巨大！！ 不好，不好





## 用 强化学习 来进行 路径规划

**强化学习的哲学启发**

* 监督学习 **优势** 在于记忆，不在于 **推断**。
* 强化学习 是一种 暴力 的推断方式。



**the setting of reinforcement learning**

* agent 处于一个 state 之中，
* agent 做了一个 action
* 对 环境 造成了 影响
* 系统返回一个 reward ，然后到达下一个 状态



**强化学习重要组成 成分：**

* Exploration , Exploitation

首先，explore in a non-greedy way，然后逐渐的变得  greedy。

找到 适合你的，一直做那个 就 OK 了。



**Q-Learning**

1. 如何更新当前 状态的 Q-Value， **one-step look ahead**
   1.  take an action，获得了一个 reward，然后到达下一个状态，
   2.  估算下个状态的 Q-value，用 greedy 的方式估算。
   3.  用 获得的 reward 和 估算的下一个 状态的 Q-value 来更新当前 Q-value！！！
2. 为什么叫 off-line：
   1. 在计算 td target 的时候用的是 $\max_{a'} Q(s',a')$
   2. 但是 simulation 的时候，用的 是用的 $a' = \varepsilon-greedy (s')$
   3. ​