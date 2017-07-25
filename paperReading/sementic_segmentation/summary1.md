# 图像语义分割总结



## Fully Convolutional Networks for Semantic Segmentation

**Key contributions**

* build a "full convolutional networks" that take input of **arbitrary size** and produce correspondingly-sized output with efficient inference and learning.

**Input**

* image of arbitrary size

**Output**

* the same size with the input image
* channel 为 类别的数量。每个 channel 代表一个类



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
* 有 multi-scale 融合的 方法。




## RefineNet

> RefineNet: generic **multi-path** refinement network that explicitly **exploits all the information** available **along the down-sampling process** to enable high-resolution prediction using **long-range** residual connections.



**key contribution**

* multi-path refinement network
* residual connections (resNet proposed it)
* chained residual pooling
* good performance



**intuition**

* repeated subsampling operations like pooling or strided convolution striding in deep CNNs lead to a significant decrease in the initial image resolution
* argue that features from all layers are helpful for semantic segmentation

  ​


**details**

* input adaptation convolution
* one ReLU in chained residual pooling



**summary**

* 整合所有 分辨率的 层，为啥最终输出 1/4, 不直接输出 1/1？？？
* this structure is similar with SSD



**my comments**

* 输出 1/1 的 score map ，不用 双线性插值
* 最后再用 crf 搞一下
* using refine-net in input
* SSD  with residual shotcuts????




**Question:**

* how to upsample within Multi-resolution Fusion



**deeplab 的两个缺点**

* computation expensive
* atrous losses important details



**为什么deeplab都是生成 1/8 的 score map， 然后用 双线性插值上采样**

* 如果生成 1/4 或者 1/1 的 score map， 内存受不了，因为 feature map 太大




## PSPnet: Pyramid Scene Parsing Network





## Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials

**key contributions**

* highly efficient approximate inference algorithm for fully connected CRF models



**details:**

* adjacency CRF structure is limited in its ability to model long-range connections within the image and smooth the object boundaries
* ​





## PyDenseCRF

```python
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

import skimage.io as io

image = train_image

softmax = final_probabilities.squeeze()

## tranpose 一下维度
softmax = processed_probabilities.transpose((2, 0, 1))

# 需要 -log(P)，这个就是求 -log(P) 的
unary = unary_from_softmax(processed_probabilities)

# The inputs should be C-continious -- we are using Cython wrapper
unary = np.ascontiguousarray(unary)

d = dcrf.DenseCRF2D(image.shape[0] , image.shape[1], 2)

# 给 DenseCRF 设定 unary potential
d.setUnaryEnergy(unary)


###################开始设定pair-wise potential ######################
# This potential penalizes small pieces of segmentation that are
# spatially isolated -- enforces more spatially consistent segmentations
# 这个是 位置 那个部分，只需要shape 足矣， sdims 表示的是 rho/gamma 参数，两维嘛，就两个一样的
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])
# compat 是 w_2 参数
d.addPairwiseEnergy(feats, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

# This creates the color-dependent features --
# because the segmentation that we get from CNN are too coarse
# and we can use local color features to refine them
# 这个 是 bilateral 那项，sdims 和上面一样，位置相关的参数，schan是 颜色强度相关的参数
feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                   img=image, chdim=2)

# compat 是 w_1
d.addPairwiseEnergy(feats, compat=10,
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)

Q = d.inference(5)

res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
```

**参考资料**
[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/)

[https://github.com/lucasb-eyer/pydensecrf](https://github.com/lucasb-eyer/pydensecrf)



## 总结

**语义分割面临的问题：**

* DCNN 的某些不变性并不是 语义分割任务所需要的
* 物体的多尺度问题
* 物体的多角度问题
* coarse segmentation problem
* 感受野问题
* 分类与定位问题 （LargeKernel matters）
* 物体的想关性（比如，船在水上）



**分辨率的降低如何解决？**

* deconv
* 插值方法 + crf 



**物体的多尺度问题如何解决？**

* 同一层，搞不同的感受野，需要层中有不同大小的核，或有不同 dilate-rate 的 atrous-conv
* 对 输入图像进行不同尺度的下采样，通过模型计算，上采样，得到的结果 fuse 一下




**coarse segmentation problem**

* refine-net，不同分辨率的特征都用到



**感受野问题的解决方法**

* 大 核，导致问题-计算量大
* atrous conv， 导致问题-计算量大
* 大 pooling kernel，小 stride，导致问题-计算量大




## 对语义分割问题的思考

* 对 anchor 点 分类的时候，需要考虑附近的 像素点 才能有可能分类正确，需要看到多大范围呢？
* pooling 在图像分类上的优势显著，但是在其它任务上就不一定 ok 了。


## 参考资料

