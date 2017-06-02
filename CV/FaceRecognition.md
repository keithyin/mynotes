# 人脸识别



**实际问题：**

* 脸的大小不一，应该怎么处理
* 光照应该怎么处理
* 年龄差距应该怎么处理
* 上亿的人脸应该怎么处理（有很多长得差不多的人）
* 如旋转不变性、尺度不变性、光照鲁棒性、甚至是对遮挡的鲁棒性等等



**人脸识别的问题：**

* intra-personal variation
* inter-personal variation

**如何把这两种特征分开？**



* Linear Discriminate Analysis
* Deep Learning for Face Recognition

## Learn Identity Features from Different Supervisory Tasks

* Face identification :classify an image into one of N identity classes
  * multi-class classification problem
* Face verification: verify whether a pair of images belong to the same identity or not
  * binary classification problem

如何使用深度学习做人脸识别：

* 找到一个非线性变换$y = f(x)$,变换之后，使得intra-person variation小，inter-person variation 大

## verification

输入一对图片，判断他们是不是同一个人。用这种方法来训练神经网络。一个人脸可以crop多个区域。



## identification

* 输入一个图片，压缩成一个160维的特征，然后再分类（10000类，一亿类呢）
* 可用于verification，和retrieval
* 当分类的类别高时，训练的会更有效。
* 一旦学习好特征之后，就可以把特征拿出来比较了。
* 比较特征可以先用PCA降维，然后用线性分类器，降维，应该降维多少？



## face 检索

如果是 Face Retrieval应该怎么搞呢？

*  identification问题，对于新加入的人脸，怎么处理，需要 zero-shot
*  identification和verification问题都可以帮助我们找到人脸的特征