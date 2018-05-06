# Person re-id 论文阅读总结



## 论文

### 一、beyond triplet loss

主要贡献是： 提出了 quadruplet loss 

解决的问题是：

* 之前的 triplet loss 只考虑了增大 inter-class variation， 并没有考虑减小  intra-class variation
* 提出的新的 quadruplet loss， 给 triplet loss 完善了一波



**我的评论**

* 感觉 工作没有啥可以批判的地方，很有道理。。。。



### 二、Learning deep context-aware features over body and latent parts for person re-id

**argue 的点是：**

* 一、传统的 CNN 最后一层输出的仅仅是高级的特征，会忽略一些小的细节，但是这些小细节又在 person re-id 的时候非常有用，比如 有没戴墨镜，鞋子是啥。
* 二、predefined rigid part based representation 的问题，可能有些 目标检测的时候没有检测好，有些图片少个胳膊少个腿啥的。 对于 predefined rigid part based representation 就有问题了。

**文中如何解决这两个问题的：**

* 第一个问题：MSCAN， 这玩意就是 inception 中的套路，只不过本文用了 dilated-conv。用了 MSCAN，就既能考虑整体，又有部分信息了。 
* 第二个问题：自动的来检测 行人的 part 而不是 直接上中下 三个 part。使用 STN 做 行人 part 的localization。这个需要仔细研究一下。



**总结：**

* 高层特征容易 忽略局部 细节信息。用 MSAN 来捕获 粗略特征和 细节特征。



**我的评论**

* 文章提出的方法依旧没有解决 缺个胳膊 少个腿的问题啊
* MSCAN 与 part-based 这两个 没有试验对比， 是哪个方法对 最终 fusion 的特征贡献大
* 序列中的人也没有对齐，是否也可以用 STN 进行对齐，然后序列 减法操作。
* 没有对齐的东西 是否都可以用 STN 预处理一下。
* STN 用到目标检测上去，如何修改？？？？



### 三、Spatial Transformer Network

既然 re-id 中看到的，就放在这里吧！

**什么是 STN**

* 是一个可微分模块
* 前向过程是 对 输入feature-map 做空间变换，然后输出空间变换后的 feature-map
* ST 机制可以分成三个部分
  * 计算变换参数（localization network）
  * 用变换参数来计算 sampling grid
  * 用 sampling grid 来采样 input feature map

**Glossary：**

* regular grid : 输出 pixel 是落在 输入 feature map 的 regular grid 上的。




**算法流程**

* 计算 变换的参数
* 根据变换参数计算 输入的 坐标（输出的 位置已知，来计算输入的坐标）
* 用计算的输入坐标对 input feature map 进行 采样。



**我的评论**

* 在小图片上可行， 大图片的话，只能在 高层特征上做文章了。



### 四、Consistent-Aware Deep Learning for Person Re-id in a Camera Network

**文章**

* a 和 b 是同一个人， a 和 c 是同一个人，那么 b 和 c 必定是同一个人，如果网络输出结果认为 b 和 c 不是同一个人，这就是有问题的。



**我的评论**

* 在 视频监控或者嫌犯追踪的时候， 什么时候 经过了 哪个 camera 才是最重要的，有道理
* 算法局限性还是挺大的吧，如果 camera 多的话，那得搞到什么时候。

**启发**

* 让 一张图片的 特征 和  序列图片的相等 做监督信号，如何？？？
* 两段视频的 关照可能不一样， 但是视频内部做差或啥啥啥，可以减去关照的影响。



### 五、 Person Re-identification in the Wild

