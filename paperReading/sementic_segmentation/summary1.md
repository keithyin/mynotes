# 图像语义分割总结



## Fully Convolutional Networks for Semantic Segmentation

**Key contributions**

* build a "full convolutional networks" that take input of **arbitrary size** and produce correspondingly-sized output with efficient inference and learning.

**Input**

* image of arbitrary size

**Output**

* the same size with the input image
* channel 为 类别的数量。每个 channel 代表一个类

**Network**



**conclusion**



**question**

* 为什么要 pad=100
* 如何给图片打标签，直接打类标签就可以了。



## SegNet

**Key Contributions**:

- Maxpooling indices transferred to decoder to improve the segmentation resolution.




## Multi-Scale Context Aggregation By Dilated Convolutions

**insight**

* Dense prediction problems such as semantic segmentation are structurally different from image classification.
* dilated convolutions support exponential expansion of the receptive field without loss of resolution or coverage.

**contribution**

* we develop a new convolutional network module that is specifically designed for dense prediction.



## 对语义分割问题的思考

* 对 anchor 点 分类的时候，需要考虑附近的 像素点 才能有可能分类正确，需要看到多大范围呢？
* ​


## 参考资料

