# Learning to Learn by gradient descent by gradient descent

开篇一句话相当经典：

> The move from hand-designed features to learned features in machine learning has been wildly successful.
>
> In spite of this, optimization algorithms are still designed by hand. In this paper, we show how the design of an optimization algorithm can be cast as learning problem.

大意是：原来的机器学习 是人工选择特征的，深度学习这块是 学习如何选择特征。对于优化方法来说，现在依旧是人工设计的，现在，我们要去学习优化方法。（有点牛逼哦）

那么关注本文的重点就是：如何将 设计优化算法的问题 转化成 学习问题。



> The performance of vanilla gradient descent, however, is hampered by the fact that it only makes use of gradients and ignores second-order information.

点出了 vanilla-gradient-descent 的问题，没有考虑 二阶导，考虑了二阶导，优化会更快



> propose to replace hand-designed update rules with a learned update rule, which we call optimizer $g$ , specified by its own set of parameters $\phi$.

用 optimizer $g$ 来替代原始的 更新 规则

原始： $\theta_{t+1} = \theta_t - \alpha_t \nabla_{\theta_t} f(\theta_t)$

本文 提出的 ： $\theta_{t+1} = \theta_t + g_t\Biggr(\nabla_{\theta_t}f(\theta_t),\phi\Biggr)$



> we will explicitly model the update rule $g$ using a recurrent neural network which maintains its own state and hence dynamically updates as function of its iterates.





> The idea of using *learning to learn* or *meta learning* to acquire knowledge or inductive biases has a long history.



> In this work we consider directly parameterizing the optimizer.

直接对 优化器进行参数化。





## Comment

* we can learn everything



## Glossary

* function of interest : 
* problems of interest: 
* learn interesting sub-structure
* inductive biases:  是对模型（参数）的一种假设，就像  Occam's razor，我们认为 模型简单 训练集效果好的网络，泛化性能好。
* optimizee： 网络（有自己的参数）
* optimizer：优化器（优化器自身也有参数）
* coordinate-wise network：