# 理解CNN 

**卷积计算具有的特点：**

* Location Invariance 
  * 如果想检测一头大象的话，拿一个大象的 filter 去卷就行了，不用关系大象在什么位置。
* Compositionality
  * 每个 filter 将低级的特征组合成高级的特征。



**Pooling 计算具有的特点**

* translation invariance（平移不变性）
* rotation invariance（[[1.5, 0], [1, 0] ] == [[0, 1.5], [1, 0]]）
* scaling invariance ([[1,1], [1,1]] == [[1, 0], [0, 0]])



**又因为图像满足这些特点，所以 CNN 完美契合图像。**



**参考资料**

[http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)



## 卷积核

**为什么使用 3×3 卷积核** : 降低计算量和参数数量

* 同样要达到 5*5 的 receptive field，一个 5×5 的卷积核 无论参数还是计算量都比两个 3×3 卷积核的要高
  * `5*5 filter` : 参数 ： 25,  计算量： 25×W×H
  * `3×3 filter`： 参数： 2* 9, 计算量： 2×9×W×H 
* 使用 两个 3×3 的还可以增强网络的表达能力，因为多了个非线性激活单元

**为什么不使用 2×2 卷积核**

* 2*2 卷积核的能力太弱，都不能表示简单的边缘检测器

**为什么使用 1×1 卷积核**

* 作用1：降低 channel 维度
* 作用2：增加卷积运算的抽象能力，相当于把原始的单层感知机换成了多层感知机。（Network In Network）



**参考资料**

[https://www.quora.com/What-is-the-significance-of-a-3*3-conV-filter-and-a-1*1-conV-filter-in-CNN](https://www.quora.com/What-is-the-significance-of-a-3*3-conV-filter-and-a-1*1-conV-filter-in-CNN)



## Pooling

* **Max Pooling**
  * 在这个区域中，有看到这个特征吗？
* **Avg Pooling**
  * 这个特征在这块区域是不是经常出现
* **K-max Pooling**
  * 在整个句子中，did you see this feature up to k times?



## CNN 的几大经典结构

**AlexNet**

* 解决了啥问题： 手工特征分类精度低的问题
* 怎么解决的：用 CNN，深度学习
* 为什么能解决：深度学习牛逼

**VGG**

* 解决了啥问题：AlexNet 分类精度低
* 怎么解决的：更好的网络结构，都是使用了 3×3 卷积核，结构更简单，更加模块化，写代码更友好
* 为什么能解决：网络更深

**GoogleNet v1**

* 解决了啥问题：一个物体，是由多个 **不同尺度的特征构成** 。考虑一个人，头，身体，所占的大小也是不同的。
* 怎么解决的：同一层中使用了 multi-scale 的 filter （3×3, 5×5）
* 为什么能解决： multi-scale 的 filter 用来捕捉 multi-scale 的特征，使得每一层的特征都混合了不同尺度的信息。
* 为什么不是解决的物体的 multi-scale 问题？ （这个部分有时间做个实验验证一下）
  * 如果解决 物体的 multi-scale 问题，应该用 不同 rate 的 dilated-conv，**并且共享权值**
  * 或者对 feature-map 进行不同 rate 的下采样。

**GoogleNet v3**

* 疯狂使用卷积核分解，更加稀疏，降低参数量



**resnet V1**

* 解决了啥问题： 人们都认为网络越深，效果越好，但是实验结果不是这样，当网络深到一定程度，再加深的话，会导致效果变差。
* 怎么解决的：增加了 residual connection
* 为什么能解决：神经网络可以学习是否跳过 residual block。这样就会得到和浅层网络一样的效果。可以解决训练过程中梯度消失的问题。

**resNext**



**DenseNet**



**SENet**





## CNN 用在 NLP 任务上的理解

*  context window modeling
* ​



# 理解 RNN

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- 针对时序信息建模
- $t$ 时刻的输出由 $t$ 时刻的输入和 $t$ 时刻的隐层状态决定 。

## LSTM

RNN 存在梯度消失的问题，梯度消失指的是 后面时刻的 `loss` 无法传到前面去，所以提出了 `LSTM`

- 增加了一条告诉公路，这条路上没有 非线性激活单元
- 只有 加法 和 乘法操作
- 一定程度上缓解了 梯度消失的问题，**没有根本上解决，因为 乘 0 就凉了**




# CNN 与 RNN 对比

**在时序信号建模中**

* cnn 需要 更少的 step 能够得到最终的表示形式
* CNN 更像是一个特征集成的一个角色，



## 参考资料

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)