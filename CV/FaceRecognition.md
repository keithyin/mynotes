# 人脸识别

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
*  