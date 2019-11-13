# Attention Is All You Need

**the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.**



**解码的时候，也是一个个 step 走的，preserve the auto-aggressive property.**



Learning **long-range dependencies** is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the **length of the paths** forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies

### Position Encoding

* 目的，使得 self-attention 的结果对位置敏感一些。

[https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#mechanisms-used-to-compose-transformer-architecture](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#mechanisms-used-to-compose-transformer-architecture)

### Multi-Head Attention

假设 $k$ 个头

* $Q, K, V$ ，分别被线性映射 $k$ 次，然后独立 attention，然后 attention 的结果 cat 起来，然后再映射。



> Multi-head attention allows the model to jointly **attend to information from different representation subspaces** at different positions.

* 一个头的也可以这样搞啊。不明白
* 不同的 attention-head 可以注意 拥有不同的注意力分布，但是一个头的就只能有一个 注意力分布了。下面的信息收集的能够增多。



# ALBERT

* 想要解决什么问题
  * 模型参数多 (GPU/TPU内存限制, 更长的模型训练时间<更多的iteration?, 前向反向慢?>)
  * 模型参数增加导致模型退化 (效果不行, resnet 的提出也是为了解决这个问题)
* 如何解决的
  * **Parameter-reduction** to lower memory consumption and increase the training speed
    * factorized embedding parameterization: decomposing the large vocabulary embedding matrix into two small matrices
    * Cross layer parameter sharing: 模型的参数数量不会随着层数的增加而增加
  * 提升模型效果: sentence-order prediction (SOP) , inter-sentence coherence,解决 下一个句子预测的效率低的问题
* 为什么能够解决
  * 由于word-embedding 目的是学习上下文无关的表示, 而 self-attention 的输出目的是学习上下文相关的表示, 所以对于 word-embedding 来说, 并不需要太大的 维度. 所以对此特征的维度进行削减. 降低模型的参数量. 
  * cross-layer parameter sharing 使得 模型的各个层完全共享参数, 减少了模型的参数量. **但是这种方法并不会减少 模型的前向和反向的计算量, 也不会减少运行时的内存消耗.**
  * sentence-order prediction: 原来的next sentence prediction 可以由两种方式解决, 一个是 topic-prediction, 另外一个才是语义连续性,  由于 topic prediction 更加容易, 所以模型可能会学到的仅仅是 topic-prediction. 这对于 理解性任务是没有啥帮助的. 所以改为 sentence-order prediction, 这个使得模型只能乖乖的学习 coherence.
* 对比其它解决方法
* 解决方法存在什么问题
* 如何改进



# ELECTRA

> GAN搞到了语言模型预训练上了.

* 想要解决什么问题
  * 预测masked token 是什么 效率低的问题,  因为每次只能训练少量的 mask
* 怎么解决的
  * 将 masked token prediction 任务改为 replaced token detection 任务
    * 使用 generatvive network 来生成 replace token, descriminator 来做 replaced token detection
    * generator 训练, maximum likelihood. discriminator也是 maximul likelihood. 两个是一起训练的, 并不是训练好一个, 再训练另一个. 
    * 细节, 如果 generator 恰巧 sample 出来真的 样本了,  就认为是 real
* 为什么能解决
  * generator 提供 hard example sampling 机制, **样本越难判断, 越得根据上下文信息进行总结归纳**
  * replaced token detection: 这玩意有啥牛逼的地方, 用了这个, 没有用 next-sentence-prediction, 说明这个玩意还能学到整个句子的含义信息.
    * replaced token detection 任务难在哪:  
      * 首先得理解句子的含义, 才能知道哪些是错的, 如果不能理解句子含义的话, 可能会搞错哪些是错的. 或者是句子纠错的任务难在哪? Mask 15% 还是保证了 大部分是正确的. 这个地方有没有 处理一下样本不均衡.
* 和其他方式的对比
* 解决方法存在啥问题
* 还有啥可以改进的地方



# Glossary

* self-attention (intra-attention):  
* symbol representation : one-hot 表示
* continuous representation :  word-embedding 表示
* position-wise : 
* At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.