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




## Semantic image segmentation with cnn and crfs

* FCN 中没有使用 dilated-conv，这导致输入图片的像素变成 1/32才能获得可观的 `receptive fields` 



**解决定位问题：**

> 因为传统的分类网络目的是用来分类，这样就引入了很多的 不变性（平移不变性，旋转不变性，由max-pool引入的），这些性质对于语义分割来说倒是劣势，因为语义分割需要 位置信息。

* 使用多层的 feature map，因为越靠近输入，位置信息保留的越多。
* employ a super-pixel representation
* 使用 crf 进行对 CNN 输出的结果进行 修正



**问题**

* unary potential, binary potential 模型是



## DeepLab:2

**key contribution**

* atrous conv
* atrous spatial paramid poolling
* crf



**detail:**

* remove the downsampling operator from the last few maxpooling layers instead `upsample the filter(atrous)`
* 去掉两层  maxpooling，  用 `atrous` 代替，最终的结果 上采样8， 得到原图尺寸的 class-map





## 总结

**语义分割面临的问题：**

* DCNN 的某些不变性并不是 语义分割任务所需要的
* 物体的多尺度问题
* 物体的多角度问题
* DCNN 过程中 分辨率下降也不是 语义分割任务所期望的



**分辨率的降低如何解决？**

* deconv
* 插值方法



**物体的多尺度问题如何解决？**

* 同一层，搞不同的感受野
* 对 输入图像进行不同尺度的下采样，得到的结果 fuse 一下







## 对语义分割问题的思考

* 对 anchor 点 分类的时候，需要考虑附近的 像素点 才能有可能分类正确，需要看到多大范围呢？
* pooling 在图像分类上的优势显著，但是在其它任务上就不一定 ok 了。


## 参考资料

