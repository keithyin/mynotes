



### Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendation

> 作用在检索模块 (retrieval)

* 解决(缓解)了啥问题

  * `without requiring fixed item vocabulary`
  * `producing unbiased estimation and being adaptive to item distribution change`
  * 推荐的item量级很大
  * 大多数 item  的用户反馈非常稀疏 (这个通过构建 user-item 交互矩阵可以看出来). 大多数 item 几乎没有用户反馈.
    * item 的反馈稀疏,  那就可以使用 content feature 来表示这个 item. 这样的话, 就一定程度上缓解了稀疏的问题. **这时候考虑的就是 哪个特征的哪个特征值的反馈比较稀疏了**
  * 如果采用 negative-sampling 训练双塔模型的话,  效率低 (需要负采样item, 然后再走一遍 item 塔). 本文使用 in-batch softmax 进行模型训练
    * **为啥不是直接 sigmoid 模型进行训练呢? 这两个有啥区别呢? 思考一下**

* 如何解决的

  * 如果使用负采样训练, 效率低下的问题

    * 采用 in-batch softmax: 对一个 mini-batch 中出现的所有 item 求 softmax
      * **因为训练的数据 是 user-item 的交互 pair. softmax 的话就是认为 用户不会和 mini-batch 中的其它 item 交互. 是不是存在这种假设????**

  * in-batch softmax is subject to sampling bias

    * correct sampling bias of mini-batch softmax using estimated item frequency

    

* 为什么能解决



* 文章片段
  * `a common recipe to handle data sparsity and power-law item distribution is to learn item representations from its content features`
    * Power-law distribution ???? 什么鬼



**疑问**

* `in-batch negatives` 表示的是什么  (一个 `mini-batch`?, 还是一天得数据构成得 `batch`?)
* 

**其它**

* 使用 content feature 可以增强系统的泛化能力  和 应对冷启动的能力

# 参考资料

[https://www.quora.com/What-does-the-concept-of-presentation-feedback-bias-refer-to-in-the-context-of-machine-learning](https://www.quora.com/What-does-the-concept-of-presentation-feedback-bias-refer-to-in-the-context-of-machine-learning)