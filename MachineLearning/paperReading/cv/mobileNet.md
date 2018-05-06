# MobileNet阅读笔记

> We present a class of efficient models called MobileNets for mobile and embedded vision applications.

`MobileNet` 主要是运行在移动端和嵌入式设备上，传统的网络一般比较大，网络参数动不动上百M。`MobileNet` 的目标是 `light weight`。



> This paper describes an efficient network architecture and a set of *two hyper-parameters* in order to build very small, low latency models that can be easily matched to the design requirements for mobile and embedded vision applications.



> MobileNets are built primarily from depthwise separable convolutions initially introduced in [26] and subsequently used in Inception models [13] to reduce the computation in the first few layers.



> Depthwise Separable Convolution : `depthwise convolution` and a `1×1 pointwise convolution`
>
> This factorization has the effect of drastically reducing computation and model size.



* Width Multiplier
* Resolution Multiplier
  * 这两个参数真心鸡肋。 





## 问题

* 为什么 depthwise separable convolution 减了那么多参数，但是效果还那么好。



## 模型压缩方法

* 分解，还有啥分解方法
*  product quantization
* hashing
* pruning 
* vector quantization
* Huffman coding
* 对训练好的网络进行 压缩。直接训练简单的网络（从结构上压缩）。
* 参数压缩，  计算量的压缩。







