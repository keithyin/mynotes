# 人脸识别



**实际问题：**

* 脸的大小不一，应该怎么处理
* 光照应该怎么处理
* 年龄差距应该怎么处理
* 上亿的人脸应该怎么处理（有很多长得差不多的人）
* 如旋转不变性、尺度不变性、光照鲁棒性、甚至是对遮挡的鲁棒性等等



**人脸识别的问题：**

* intra-personal variation
* inter-personal variation

**如何把这两种特征分开？**



* Linear Discriminate Analysis
* Deep Learning for Face Recognition

## Learn Identity Features from Different Supervisory Tasks

* Face identification :classify an image into one of N identity classes
  * multi-class classification problem
* Face verification: verify whether a pair of images belong to the same identity or not
  * binary classification problem

如何使用深度学习做人脸识别：

* 找到一个非线性变换$y = f(x)$,变换之后，使得intra-person variation小，inter-person variation 大

## verification

输入一对图片，判断他们是不是同一个人。用这种方法来训练神经网络。一个人脸可以crop多个区域。



## identification

* 输入一个图片，压缩成一个160维的特征，然后再分类（10000类，一亿类呢）
* 可用于verification，和retrieval
* 当分类的类别高时，训练的会更有效。
* 一旦学习好特征之后，就可以把特征拿出来比较了。
* 比较特征可以先用PCA降维，然后用线性分类器，降维，应该降维多少？



## face 检索

如果是 Face Retrieval应该怎么搞呢？

*  identification问题，对于新加入的人脸，怎么处理，需要 zero-shot
*  identification和verification问题都可以帮助我们找到人脸的特征



## faceNet

**创新点：**

* 直接优化人脸图片的embedding，而不是原来人们用的神经网络的中间层
* harmonic embeddings（和谐的 嵌入？？）
* harmonic  triplet loss
* 直接学习欧式 embedding，我们可以学习其它距离？
* triplet based loss function
* ​

**数据处理：**

* triplet 
  * 包含两个匹配的人脸 thumbnail, 和一个不匹配的人脸 thumbnail
  * thumbnail are tight crops of face area, 就切脸那部分（怎么切的？）
  * 切完的 thumbnail， 可能会进行  放缩（scale），变换（translation）
  * 如何选择 triplet很重要
    * 提出了一个新颖的在线 **负样本** 挖掘策略， 保证随着训练的进行，triplet的可分辨性越来越难
    * 什么样的策略？ 神奇的很
* 如何选择 triplet?
  * 目的：给定 $x_i^a$ 找到，$x_i^p=\text{argmax}_{x_i^p}||f(x_i^a)-f(x_i^p)||_2^2$  和$x_i^n=\text{argmin}_{x_i^n}||f(x_i^a)-f(x_i^n)||_2^2$
  * 计算所有数据集的$\text{argmax}$ 和$\text{argmin}$是不可行的，那该怎么办？
    * 每n步，用最近的checkpoint，生成一次triplets，然后在子集上计算$\text{argmax}$ 和$\text{argmin}$.   在子集上计算是什么意思？
    * 在线生成triplets。 从 mini-batch中选择 $\text{argmax}$ 和$\text{argmin}$。

**训练：**

* 选用两个网络：ZFnet  或者 Inception(2014)



**测试：**

* 为了提高聚簇的准确性
  * 开发了一个 hard-positive 挖掘算法，`对于同一个人的embedding，鼓励球体聚簇`

**尚不明了的东西:**

* 什么是 curriculum learning?
* collapsed model



**想法:**

* 在加上人脸的其它手工特征，一起放入 `neural network` 是不是效果会更好。
* ​




## Deep Face Recognition



##  Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

* detection & alignment



**创新点:**

* 提出了一个框架，将`detection`和`alignment`搞到了一起
* 使用了级联`CNN`
* 提出的框架包含了三个阶段
  * 浅层CNN快速生成候选窗口
  * 然后，通过将第一步产生的窗口通过一个比较复杂的CNN映射来 调整window
  * 最后，用一个更牛逼的CNN调整结果，最终输出5个人脸关键点。
* 提出一个方法用于创建 合理的人脸数据，只需要有限的人类标注
* ​