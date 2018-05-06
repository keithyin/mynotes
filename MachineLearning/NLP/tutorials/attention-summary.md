# NLP 常用的各种 Attention 及其pytorch实现

## `LuongAttention` 

来自论文[Effective Approaches to Attention-based Neural Machine Translation](http://cn.arxiv.org/pdf/1508.04025.pdf)



**dot**
$$
a_t(s) = \frac{\exp h_t^T  \bar h_s}{\sum_{s'} \exp h_t^T \bar h_{s'}}
$$
**general**
$$
a_t(s) = \frac{\exp h_t^T  W_a \bar h_s}{\sum_{s'} \exp h_t^T W_a  \bar h_{s'}}
$$
**concatenate**
$$
a_t(s) = \frac{\exp W_a [h_t^T ; \bar h_s]}{\sum_{s'} \exp W_a [h_t^T ; \bar h_{s'}]}
$$




**求得了分布之后**
$$
\text{attn_val} = \sum_{s'} a_t(s') \bar h_{s'}
$$




**scaled dot-product attention**
$$
\text{Attention(Q,K,V)} = \text{softmax}(\frac{QK^T}{\sqrt d_k})V
$$

> scale 的目的控制 $QK^T$ 值的方差， 如果 $Q, K$ 方差都为 1, 那么 scale 之后的方差依旧为 1



**multi-head attention**





**self-attention**