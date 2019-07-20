# Deep Recurrent Q-Learning Summary

对使用  RNN 来编码 observation 来做总结。

* Deep Q-Learning with Recurrent Neural Networks
* Deep Recurrent Q-Learning for Partially Observable MDPs
* Deep Attention Recurrent Q-Network



## Deep Q-Learning with Recurrent Neural Networks



### 论文阅读

> we investigate these architectures to overcome the difficulties arising from learning policies with long term dependencies.

RNN 有很好的解决长时间依赖数据的 特性。



> some state-of-art perform poorly on several games that require long term planning. 

一些 state-of-art 的方法对于需要长时间 plan 的问题 效果不好。为什么会不好呢？？？？？



> In practice, DQN is trained using an input consisting of the last four game screens. Thus, DQN performs poorly at games that require the agent to remember information more than four screens ago.

传统 DQN，只用当前 **四帧** 的 observation 来作为 state，如果当前的 state 需要更久之前的 observation，那么 DQN，就歇菜了。

但是感觉 RNN，只能更好的 编码 **状态（state）**，但是并不能更好的帮助 做 long-term planning 啊。感觉 RNN：

* 对过去很友好
* 对 long-term planning 没有啥帮助
* 还会指数级的增加 state 的数量。

如果想要 long-term planning 应该更多的去关注 未来，而不是过去！！

感觉这地方还可以这么思考： MDP 可以帮助我们解决 credit-assignment 问题，但是如果 真正的重要的 $(s,a)$ pair 离 最终获得 credit的 时间，相距很远，那么算法是否还能正确的 credit-assignment。



> The advantage of using attention is that it enables DRQN to focus on particular previous
> states it deems important for predicting the action in the current state.

这么搞的话，数学模型 MDP 不就崩溃了？？？？？



## Deep Recurrent Q-Learning for Partially Observable MDPs

### 论文阅读

> adding recurrency to a DQN by replacing the first post-convolutional fully-connected layer with a recurrent LSTM.



> DQN will be unable to master games that require the player to remember events more distant than four screens in the past.

意思就是：如果 player 需要更多的 帧 来搞清楚自己所处的状况（i.e. state），那么 原始 DQN 就不好使了。至于到底需要多少帧 来搞清楚自己当前的状况？ 不清楚，好，那就用 RNN 来搞定吧。









### Comment

* 为啥不把 过去的 action 也考虑进来呢？ 如果想考虑的话，应该怎么建模比较好

