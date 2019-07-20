# Pointer Network

**Pointer Network 可以干什么?**

compute **variable-length** probability-distributions with a **fixed** architecture, over **arbitrary inputs**.



**fixed architecture**

* 如果直接使用 Recurrent Neural Network 来计算 变化长度 的概率分布的话, 那么结构一定不可能是固定的, 因为 Recurrent Neural Network unroll 的长度不同. 
* Pointer Network 如何做到的呢?



**如何做到的?**

* 使用 Attention 机制, 因为 Attention 会计算 输入的分布, 这个分布就是 **variable-length** probability-distributions
* Recurrent Neural Network 仅仅作为一个载体, 如果对一个输入需要两个 概率分布, 那么 RNN unroll 的长度就设置成2 . 这个 2 和 输入的长度是没有关系的, 所以称之为 **Fixed Architecture** . 



## 参考资料

[https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264](https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264)

[https://www.zhihu.com/search?type=content&q=Pointer%20Network](https://www.zhihu.com/search?type=content&q=Pointer%20Network)