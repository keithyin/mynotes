# 理解 CNN



**卷积计算具有的特点：**

* Location Invariance 
  * 如果想检测一头大象的话，拿一个大象的 filter 去卷就行了，不用关系大象在什么位置。
*  Compositionality
  * 每个 filter 将低级的特征组合成高级的特征。



**Pooling 计算具有的特点**

* translation invariance（平移不变性）
* rotation invariance（[[1.5, 0], [1, 0] ] == [[0, 1.5], [1, 0]]）
* scaling invariance ([[1,1], [1,1]] == [[1, 0], [0, 0]])



**又因为图像满足这些特点，所以 CNN 完美契合图像。**



## 参考资料

[http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

