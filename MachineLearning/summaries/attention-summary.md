# attention总结



## 自然语言处理中常用的attention

可以抽象为三个元素 `<Q, K, V>`

* `<Q, K>` 计算 `alignment weight`
* `alignment weight` 与`V` 计算加权和



**BahdanauAttention**

<img src="/Users/yinpeng02/Documents/gitspace/mynotes/MachineLearning/imgs/bah-attention.png" alt="bah-attention" style="zoom:50%;" />

在预测时刻`t` 的输出的时候

* 计算 decoder  $s_{t-1}$ 与 encoder所有时间步的隐状态的 score
* 用 score 作为权重，计算 encoder所有隐状态向量的加权和
* 加权和与 $s_{t-1}, y_{t-1}$ 一起送入 `rnn_cell` 做当前步的预测

**LuongAttention**

在预测时刻`t` 的输出的时候

- 计算 decoder  $s_t$ 与 encoder所有时间步的隐状态的 score
  - 计算 score 的方式有多种
- 用 score 作为权重，计算 encoder所有隐状态向量的加权和
- 加权和 与 $s_t$ 用于计算当前时间步的输出
  - 注意：加权和 是不经过 rnn_cell 的，这是和 Bahdanau 的不同
- Local 版本
  - 只考虑一个 window 内的内容



**SelfAttention**

* 一句话的 embedding 向量映射程 三个 embedding 向量
  * 分别为 $<Q, K, V>$ 然后进行计算 score，然后加权和
* 用在 encoder 部分，因为所有的 token 已知



**MonotonicAttention**





## CV中的 Attention



## 其它

