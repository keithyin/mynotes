# cnn for modeling sentences

**semantic modeling of sentences**

* DCNN (Dynamic cnn)
* Dynamic $k$-max pooling
* Doesn't rely on parse tree



## Dynamic $k$-max pooling

>  First, k-max pooling over a linear sequence of values returns the subsequence of k maximum values in the sequence, instead of the single maximum value. Secondly, the pooling parameter $k$ can be dynamically chosen by making $k$ a function of other aspects of the network or the input.

即：使用 max 排序，然后取前 $k$ 大的值。



