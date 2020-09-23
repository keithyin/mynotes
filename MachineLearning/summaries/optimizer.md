对 optimizer.pdf 的补充解释



* 大batch-size 与 小 batch-size

  * 如果训练样本非常多，且非常符合真实的样本分布，感觉还是大batch-size比较好
  * 但是：样本量一般是不够的：
    * 大batch-size，更容易拟合数据集，更容易陷入sharp-minima。
    * 小batch-size由于引入了更大随机性，所以容易陷入smooth-minima

* Learning Rate：

  * 大 learning rate 模型容易不收敛
  * 小leaning rate，模型可能进到一个 local-minima 就出不来了。容易陷入局部最小值
  * 一般策略是learning rate decay

# 模型参数初始化

好的初始化标准：

1. 激活值大部分都被激活。不然优化个毛毛
2. 所有层输入的值都是0均值， 方差是多少？当然是1比较好。



  * uniform(0,1)
  * gaussian(0,1)
  * xavier_uniform_
      * 希望每层的输入均值都是0均值1方差，且每层的梯度也是0均值1方差
      * nVar(w) = 1 (然后根据用的是均匀分布还是高斯分布来算出对应的分布参数)
      * 考虑到正向和反向，n 一般设置为 $\frac{(fin+fout}{2}$
      * 适用于 tanh 激活函数。
  * xavier_normal_
  * kaiming_uniform_
      * 适用于 relu and sigmoid, 因为 relu 是有一半的概率不激活的， 所以relu的方差计算是
      * $\frac{fin+fout}{4} Var(w) = 1$ 然后根据该值来确定 均匀分布或者高斯分布的 分布参数
  * kaiming_normal_

# 输入特征处理







# 模型优化算法

https://ruder.io/optimizing-gradient-descent/index.html#batchgradientdescent

* SGD

  * 存在的问题：
    * 学习率每个参数是统一的
    * 学习率不好选，大了震荡，小了局部最优(鞍点)
* SGD+momentum

  * 解决如果loss的等高面非常椭圆，优化导致的震荡问题。将之前的梯度加起来，避免左右摇摆。
  * 存在问题：如果原来不存在左右摇摆问题，那么训练几轮之后，最终的梯度会非常大。
* Nesterov

  * 解决：在再次碰到斜率上升时能够稳住，而非像momentum一样，一股脑充到斜率上升方向，再回来。
  * 算法：用动量来更新参数时，先看一下如果用当前的动量更新，求出来的导数是啥鸟样，然后用这个导数来修正 动量的值。如果用当前动量更新之后求到的导数和原始动量是同方向，会进行梯度增强，否则，会进行梯度抵消。
* AdaGrad：适合处理稀疏数据

  * 思想：出现次数较少的特征使用更大的学习率。出现次数多的，使用小的学习率。
  * 对每个参数进行学习率修正：如何进行学习率修正呢？
    * 使用历史累积的梯度平方+当前时刻的梯度进行调整。（出现少的特征，历史的梯度就是不多咯。这样可以保证高学习率）
  * 存在的问题：由于累积的是梯度的平方，这值是递增的。不大好
* RMSProp

  * AdaGrad是累加历史所有+当前时刻，RMSProp：是累加时间窗口内的梯度平方。时间窗口累加使用Moving Average算法来操作。
* AdaDelta：修正AdaGrad学习率一直递减的问题

  * AdaGrad是累加历史所有+当前时刻，AdaDelta：是累加时间窗口内的梯度平方。时间窗口累加使用Moving Average算法来操作。
* 这算法看起来似乎挺复杂。。。。
* Adam （这个可能需要推导一下了。）
  * 算法：
    * 历史梯度的moving average （动量）
    * 历史梯度平方的 moving average （这个是为了模仿 经常出现的梯度低，不经常出现的梯度高）
    * 融合了RMSProp + momentum

