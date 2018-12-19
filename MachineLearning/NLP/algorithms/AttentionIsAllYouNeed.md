# Attention Is All You Need

**the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.**



**解码的时候，也是一个个 step 走的，preserve the auto-aggressive property.**



Learning **long-range dependencies** is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the **length of the paths** forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies



## Position Encoding

* 目的，使得 self-attention 的结果对位置敏感一些。

[https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#mechanisms-used-to-compose-transformer-architecture](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#mechanisms-used-to-compose-transformer-architecture)





## Multi-Head Attention

假设 $k$ 个头

* $Q, K, V$ ，分别被线性映射 $k$ 次，然后独立 attention，然后 attention 的结果 cat 起来，然后再映射。



> Multi-head attention allows the model to jointly **attend to information from different representation subspaces** at different positions.

* 一个头的也可以这样搞啊。不明白
* 不同的 attention-head 可以注意 拥有不同的注意力分布，但是一个头的就只能有一个 注意力分布了。下面的信息收集的能够增多。



## Glossary

* self-attention (intra-attention):  
* symbol representation : one-hot 表示
* continuous representation :  word-embedding 表示
* position-wise : 
* At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.