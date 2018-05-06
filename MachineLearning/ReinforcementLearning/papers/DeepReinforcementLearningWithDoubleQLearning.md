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

--------------------

Double Q learning is a fix for a problem observed in Q learning, especially in Q-Learning with function approximation: estimates of Q value are noisy, and this makes return estimation biased.

Let's say true return value is 0 for all actions at the current state. But because of the noise in estimation, some of actions may get e.g. small positive values, and other actions can get small negative values (let's say +0.05 and -0.05). In Q-learning return estimate is computed using Q function: we evaluate Q function for all possible actions in this state, and choose an action with a largest Q value. But because of a noise maximum will be a small positive value (0.05), not zero, and it will happen every time. So the estimate in vanilla Q learning is biased.

But if we use another noisy Q2 function instead of Q to select a best action, then in case of noise we can get either small positive or a small negative value (assuming this Q2 is noisy in a different way), so on average the value will be closer to 0 - the estimate becomes unbiased.

-------------------------------

## 参考资料

[https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/](https://www.reddit.com/r/MachineLearning/comments/57ec9z/discussion_is_my_understanding_of_double/)

