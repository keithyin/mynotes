#Sampled softmax

* https://www.tensorflow.org/extras/candidate_sampling.pdf
* https://stats.stackexchange.com/questions/362311/candidate-sampling-for-softmax-tensorflow-sampling-probability

* https://towardsdatascience.com/hierarchical-softmax-and-negative-sampling-short-notes-worth-telling-2672010dbe08
* https://ruder.io/word-embeddings-softmax/

### candidate sampling

* 什么是 candidate sampling
  * 对于分类任务,训练时候的类别由 当前**真实的类别**和**采样出来的负例**构成新的类别集合, 在这个类别集合上进行 softmax_cross_entropy 或者 sigmoid_cross_entropy
* 为什么使用 candidate sampling
  * 如果分类的类别过多, 那么在计算 partitaion function 的时候计算资源要求极大. 如果使用 candidate sampling, 可以极大程度减少一次前向和反向的时间

以下介绍一些 candidate sampling算法



**几个数学符号说明**:

* $Q(y|x)$: candidate sampling 分布, 给定 $x$ , 采出 $y$ 的概率
* $K(x)$ : 一个不依赖 类别的函数, 由于 softmax 会进行一个 normalization 操作, 加上这么一个值对计算出来的概率分布没有啥影响.

###Sampled Softmax

> * 一个快速计算 softmax 分类器的方法
>
> * single-label 问题

定义: $F(x,y)\leftarrow \log(P(y|x))+K(x)$

* $P(y|x)$ : 给定上下文 $x$ , 预测出来的类别为 $y$ 的概率是多少





### Noise Contrastive Estimation



### Negative Sampling



### Sampled Logistic



### Full Logistic







 