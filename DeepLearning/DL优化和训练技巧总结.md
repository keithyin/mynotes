# 深度学习优化方法和训练技巧总结

# 1. dropout

来自于论文 Improving neural networks by preventing co-adaptation of feature detectors.

什么叫co-adaptation?

* When you build a neural network with certain number of neurons in its hidden layers, you ideally want each neuron to operate as an independent feature detector. If two or more neurons begin to detect the same feature repeatedly (co-adaptation), the network isn't utilizing its full capacity efficiently. It's wasting computational resources computing the activation for redundant neurons that are all doing the same thing.



**使用dropout可以强制同一层不同的神经元(hidden unit)去学习更加robust的特征。**



**L2 constraint**

* 每个神经元(unit)的权值的L2范数不能超过某个值，如果超过的话，$w = w/l2\_norm(w)$
* L2范数的值设置为$\sqrt{15}$, 为什么呢？



## 2. ReLU

来自于论文：Delving Deep into Rectifiers

神经网络在目标识别领域卓越的效果取决于以下因素：

* 网络复杂度额增加（更深、更宽）
* 新的激活函数
* 复杂的层设计（例如：inception）

更好的泛化性能来自于：

* 有效的正则化方法
* 数据增强技术
* 大规模数据



本文认为ReLU在模型精度的增长过程中也扮演这非常重要的角色。