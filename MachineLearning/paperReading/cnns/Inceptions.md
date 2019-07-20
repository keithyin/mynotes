# 关于几篇Inception的论文总结

## Going Deeper with Convolutions
1. 卷积层采用了不同大小的卷积核,使得对物体的大小更加不敏感
2. 采用1*1卷积核降低维度(减少参数数量)

## Rethinking the Inception Architecture for Computer Vision
**背景:**
1. 数据量足够大的时候,网络越深,越宽,效果越好.
2. 但是网络的深度代表参数的个数变多,不利于在手机上使用

**解决的问题:**
1. 构建了一种网络,虽然很深,但是参数很少

**使用的方法:**
1. 分解卷积的方法降低参数

**网络设计方法:**
1. 避免网络的表示瓶颈,每层都应该考虑 (one should avoid bottlenecks with extreme compression.)
2. In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand.
3. 在神经网络中,高维表示可以容易的进行局部处理(为什么呢?).(就像图片,是个高维数据, 可以进行局部处理,卷积运算久居右这样的性质)
4. 在低维embedding上进行空间聚集只会损失少量信息或不损失信息.(例如:before performing a more spread out (e.g 3*3) Convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects.) 这里的 spread out 我感觉指的是depth上的spread out, reduce the dimension of input 感觉也是depth上的reduce, 所以关系是 reduce-depth --> spatial --> spread-out depth????
如何解释这个原因? 作者的假设是(the **correlation between adjacent unit** results in much less loss of information during dimension reduction, if the results are used in a spatial aggregation context.)
5. 平衡网络的宽度与广度.(Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network!)

**碰到的问题:**

* 前几层使用 卷积核分解的方法 会使性能降低,但是在feture map的size是(12*12 到 20*20)的时候使用这种分解的方法会很好.
* 导致上述结果的原因是什么????


**提出了一个平滑的损失函数用来增强网络的泛化能力**

**结果怎么样:**
## Inception-v4, inception-resnet and the impact of residual connections on learning

**将Inception模块与residual connections联合起来使用**
好处：
* 加速了Inception网络的训练过程,并不同意`He`说的，引入残差层会对深度的网络有非常重要的作用。
* 提高网络的准确性

问题：
* 但单一模型性能的提升并没有导致集合模型性能的大幅度提升
* 如果filter的数量过多的话，网络结构不够稳定，可以通过放缩残差的值来解决这个问题
* if the number filters exceeded 1000, the residual variants started to exhibit instabilities and the network has just "died " early in the training, meaning that the last layer before the average pooling started to produce only zeros after a few tens of thousands of iterations.

* This could not be prevented, neither by lowering the learning rate, nor by adding an extra batch-normalization to this layer.

* They found that scaling down the residuals befor adding them to the previous layer activation seemed to stabilize the training.  the scale factor is 0.1 and 0.3.

* They found that if the number of filters is very high, then even a very low(0.00001) learning rate is not sufficient to cope with the instabilities and the training with high learning rate had a chance to destroy its effects. They found it much more reliable to just scale the residuals.

* 为什么要scale residual呢？ net += scale*residual ，然后再送入activation_fn。 net里面的值是 大于等于 0的， residual 是 负无穷到正无穷的。

**对比的模型：**
* Inception-v3 VS Inception-ResNet-v1
* Inception-v4 VS Inception-ResNet-v2


**reidualal与non-residual之间的区别：**

* residual: 仅在`traditional layers`上使用 `BN`， 不在`summation`上使用，目的是，减少GPU内存损耗。
 


## 遗留问题?
1. 为什么前几层不用`Inception`模块?
内存原因
2. patch-alignment issues?

## 我的问题

* 导致网络不稳定的因素有哪些
