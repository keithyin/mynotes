# Meta Learning

> A key aspect of intelligence is versatility – the capability of doing many different things. 



当前的 AI 系统可以从头学习一个非常复杂的技巧，通过大量的数据和训练。但是，如果我们希望我们的 agents 能够学习很多技巧，然后将这些技巧调整到其他的环境中（这些环境中，我们无法提供足够的数据来让 agents 从头学习）。

这样，我们就需要 agents 能够通过从以前环境中学习到的东西来快速的适应新的环境。



实现这中效果的途径就是 Meta Learning，也叫 Learning to learn，即：学习如何学习（Learning how to learn）



**Meta Learning 要解决的问题**

* 能够有效的利用之前任务的学习到的知识
* 将 agents 快速的调整到新的 环境中



**如何实现 Meta Learning**

* 学习更好的优化算法（Learning Optimizers），取代 （Adam，SGD）
* 学习 初始化参数（fine-tune）
  * 这种方法有个问题：向小数据集上转型时，会过拟合。
* 学习网络超参数
* Metric Learning： （学习一个好的 metric 的空间）



一般 Meta Learning 都有两个网络：Meta-Learner，Learner。

* Meta-Learner: Meta-Learning 的心脏
* Learner：task-specific 的网络
* Meta-Learner 指导 Learner 学习













## 参考资料

[http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)

[https://chatbotslife.com/why-meta-learning-is-crucial-for-further-advances-of-artificial-intelligence-c2df55959adf](https://chatbotslife.com/why-meta-learning-is-crucial-for-further-advances-of-artificial-intelligence-c2df55959adf)

