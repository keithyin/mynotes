# Double Q-Learning



**Result**

* 减少了 overestimate action values 的问题
* 比 DQN 在某些游戏上得到了更好的结果



**Overestimate Action Values under certain conditions!!!**



**什么是 Overestimate Action Values**



**为什么会出现这个问题**

> The max operator in standard Q-learning and DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in over-optimistic value estimations.

为什么 values both to select and to evaluate an action 就会导致 Overestimate Action Values 问题呢？



**怎么解决这个问题**



**为什么能解决这个问题**



## 参考资料

[https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/](https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/)

