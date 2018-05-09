# Attention Is All You Need

**the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.**



**解码的时候，也是一个个 step 走的，preserve the auto-aggressive property.**



Learning **long-range dependencies** is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the **length of the paths** forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies



## Glossary

* self-attention (intra-attention):  
* symbol representation : one-hot 表示
* continuous representation :  word-embedding 表示
* position-wise : 
* At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.