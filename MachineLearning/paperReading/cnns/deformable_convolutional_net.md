# deformable convolutional networks

**motivation**

* the receptive field size of all activation units in the same CNN layer are the same
* *max-pooling for small translation-invariance*


* lacks internal mechanisms to handle the geometric variations



**implementation**

* Adding 2-D offsets to the regular sampling grid in the standard convolution
* the offsets are learned from the preceding feature maps.
* 原始卷积是 3×3 ，deformable 之后还是  3×3 ，参数的位置跑了 



**note**

* 如果保证 feature map 的上的单元感受野不同
* 用什么特性来保证感受野不同的（因为 feature map 上的感受野最好与输入图像相关）
* 旋转不变性可以添加上去，如何增加角度的不变性（正脸，侧脸啥的都不影响）
* 对 input feature map 的采样更加随意了，还能搞出啥高明的采样方法？
* 位移是整数，这个整数怎么搞出来。(无所谓小数，使用双线性内插来取值。)