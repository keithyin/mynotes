# Actor-Mimic

**The ability to act in multiple environment and transfer previous knowledge to new situations can be considered a critical aspect of any intelligent agent.**

* act in multiple environment
* transfer previous knowledge to new situations



本文贡献：

> define an autonomous agent to **learn how to behave in multiple tasks simultaneously**, and **generalize its knowledge to new domains**. 



**Actor-Mimic**

>  exploits the use of **DRL** and **model compression** techniques to train a **single policy network** that learns how to act in a set of distinct tasks **by using the guidance of several expert teachers.** 

* expert teachers ???



> Although the DQN maintains the same network architecture and hyper-parameters for all games, the approach is limited in the fact that **each network only learns how to play a single game at a time.**

* DQN 的局限性在于，对于每个游戏，都需要训练一个 agent



> A network trained to play multiple games would be able to generalize its knowledge between games, achieving a single compact state representation as the inter-task similarities are exploited by the network.
>
> Having been trained on enough source tasks, the multi-task network can also exhibit transfer to new target tasks, which can speed up learning.

* 用一个 multi-task 训练的 agent 进行 transfer learning



**Multi-Task Learning : Actor-Mimic**

> Actor-Mimic that leverages techniques from model compression to train a **single multi-task network** using guidance from a set of game-specific expert networks.



## Actor-Mimic

* Source games : $S_1, S_2, ..., S_N$
* guidance from a set of expert DQN networks $E_1, E_2, ..., E_N$



**数据如何来**

* ...



**Objective**
$$
\mathcal L^i_{Actor-Minic}(\theta,\theta_{f_i}) = \mathcal L^i_{policy}(\theta)+\beta\mathcal L^i_{FeatureRegression}(\theta,\theta_{f_i})
$$

* do as i do
* i do it because



## Transfer

**Actor-Mimic As Pre-training**

用上面的方法训练好一个 AMN（Actor-Mimic Network）

用 AMN 的参数初始化 新的 DQN。然后放在 target-task 上训练。



**深度学习用在 RL 上，就把它看作一个特征提取器就好。**



## 参考资料

[https://medium.com/@shmuma/summary-actor-mimic-deep-multitask-and-transfer-reinforcement-learning-dd2f24b441e1](https://medium.com/@shmuma/summary-actor-mimic-deep-multitask-and-transfer-reinforcement-learning-dd2f24b441e1)

