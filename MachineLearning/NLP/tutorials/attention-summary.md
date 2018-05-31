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



**[location sensitive attention](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf) **

* adding location awareness to the attention mechanism
* avoid concentrating the attention on a single frame
* content based ： $\alpha_t = Attend(h_t, e_t)$
  * 这个方法的问题在与， 如果 $e_a, e_b$ 值一样，无论他俩的位置多远，都会赋予相同的 attend weight
* location based：$\alpha_i = Attend(h_t, \alpha_{t-1})$
* hybrid : $\alpha_t = Attend(h_t, e_t, \alpha_{t-1})$