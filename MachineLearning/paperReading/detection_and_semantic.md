# Fully Convolutional Networks for Semantic Segmentation

**Glossary:**

* input stride: atrous algorithm, dilation
* output stride: 经常提到的stride就是 output stide
* skip atchitecture: 有跳跃连线的网络，就像resnet
* shift and stitch: 移位一次，计算一次输出，然后把不同移位的输出粘起来。但是如何shift呢？我怎么感觉交换更好
  * 每f个像素进行循环，直到循环一遍。
* in-network upsampling: 用 `deconv`

**Key insight:**

* To build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning.

**Note:**

* Global information resolves what *while* local information resolves where.
* Deep, coarse, semantic information and shallow, fine, appearance information.
* using deconvolution layers for upsampling.(efficient and effective alternative)

**How to connect coarse output to dense pixels:**

* interpolation
* deconvolution
* shift and stitch??? 没搞明白




**convnets:**

* local tasks with structed ouput.(bounding box, Segmentation)

**Q&A:**

* The classification nets subsample to keep filter small and computional requirements reasonable, why?
可能解释：下采样可以很快的使filter看到global的信息。但是这样也会丢失信息呀。如何两者兼顾


# Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs


# YOLO(You only look once)

* For object detection

## procedure

* divides the input image into an S*S *grid*, if the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
* each *grid cell* predicts B *bounding boxes* and confidence score for those boxes.
  * 预测的B个bounding box的置信度保存在哪？
* Each *bounding* box consits of 5 predictions: x,y,w,h
