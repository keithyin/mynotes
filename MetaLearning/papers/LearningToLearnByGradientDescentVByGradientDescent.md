# Learning to Learn by gradient descent by gradient descent

**key word**

* meta-learning
* learning to learn




## 文章阅读

**Abstract**



开篇一句话相当经典：

> The move from hand-designed features to learned features in machine learning has been wildly successful.
>
> In spite of this, optimization algorithms are still designed by hand. In this paper, we show how the design of an optimization algorithm can be cast as learning problem.

大意是：原来的机器学习 是人工选择特征的，深度学习这块是 学习如何选择特征。对于优化方法来说，现在依旧是人工设计的（SGD，Adam，。。。），现在，我们要去学习优化方法。（有点牛逼哦）

那么关注本文的重点就是：如何将 设计优化算法的问题 转化成 学习问题。



**Introduction**

> The performance of vanilla gradient descent, however, is hampered by the fact that it only makes use of gradients and ignores second-order information.

点出了 vanilla-gradient-descent 的问题，没有考虑 二阶导，考虑了二阶导，优化会更快





> specialization to a subclass of problems is in fact the *only* way that improved performance can be achieved in general.

专门解决一个子问题 是**提升性能**的唯一方法。



> propose to replace hand-designed update rules with a learned update rule, which we call optimizer $g$ , specified by its own set of parameters $\phi$.

用 optimizer $g$ 来替代原始的 更新 规则

原始： $\theta_{t+1} = \theta_t - \alpha_t \nabla_{\theta_t} f(\theta_t)$

本文 提出的 ： $\theta_{t+1} = \theta_t + g_t\Biggr(\nabla_{\theta_t}f(\theta_t),\phi\Biggr)$



> we will explicitly model the update rule $g$ using a recurrent neural network which maintains its own state and hence dynamically updates as function of its iterates.





> The goal of this work is to develop a procedure for constructing a learning algorithm which performs well on a particular class of optimization problems. 
>
> Casting algorithm design as a learning problem allow us to specify the class of  problems we are interested in through example problem instances.

通过 事例问题实例 来创建一个对应某类问题的优化算法。（learning algorithms 指的是啥？ 优化算法，还是机器学习算法）



> The idea of using *learning to learn* or *meta learning* to acquire knowledge or inductive biases has a long history.



> In this work we consider directly parameterizing the optimizer.

直接对 优化器进行参数化。







## Comment

> A classic paper in optimization is ‘No Free Lunch Theorems for Optimization’ which tells us that no general-purpose optimization algorithm can dominate all others. So to get the best performance, we need to match our optimization technique to the characteristics of the problem at hand





* we can learn everything




要理解这篇文章就要知道什么是 NFL 定理：

* 没有一种 优化算法可以适用与所有任务

就像我们拥有的一些优化算法：RMSprop，ADAM，AdaGrad。。。他们不会适用所有的任务的。

还有一个问题就是：这些算法是人类设计的（就像初期的机器学习算法，特征提取是人类设计的一样），我们能不能直接学习出来 参数的更新的方法。

本文的目的就是学习出一个 **某类问题** 的参数更新方法。 



> Learning how to learn

how to learn : 在这篇文章中指的是如何更新网络参数

Learning to learn : 学习  如何更新网络参数。





## inductive bias (归纳偏好)

> 机器学习算法在学习过程中对某种类型假设的 **偏好** ，称为 **归纳偏好（inductive bias）** （i.e. 机器学习算法更偏向与哪种参数。）

> 任何一个有效的机器学习算法必有其归纳偏好，否则他将被假设空间中看似在训练集上 **等效** 的假设所迷惑，从而无法产生确定的学习结果。（学习结果？ 难道是参数？）

> 归纳偏好可看作学习算法自身在一个可能很庞大的假设空间对假设 **进行选择** 的启发式或 **价值关** （i.e. 可能很多假设都能满足训练集，那么到底选择那个假设呢？ 这就根据 学习算法的 **归纳偏好** 来搞定）



> 归纳偏好对应了学习算法本身所做出的关于 **什么样的模型更好的假设**



> 学习算法的归纳偏好与问题是否相配，往往会起到决定性的作用。





**有没有一般性的原则来引导算法确立正确的偏好呢？**

* meta learning 来了。。。。
* 奥卡姆剃须刀 是一个原则






## Glossary

* function of interest : 
* problems of interest: 
* learn interesting sub-structure
* inductive biases:  是对模型（参数）的一种假设，就像  Occam's razor，我们认为 模型简单 训练集效果好的网络，泛化性能好。
* optimizee： 网络（有自己的参数）
* optimizer：优化器（优化器自身也有参数）
* coordinate-wise network：