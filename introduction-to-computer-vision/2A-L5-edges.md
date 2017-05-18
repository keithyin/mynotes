# Edges in image (Edge detection)

* surface normal discontinuity(表面上的不连续)
* depth discontinuity(深度上的不连续)
* surface color discontinuity(表面颜色的不连续)
* illumination(shadow edge)

## Edge Detection

* Basic idea: look for a neighborhood with strong signs of change
  * Problems: neighborhood size? how to detectt change?

## how to detect change

**Derivatives and edges**

**edges correspond to `extrema` of derivatives (positive or negtive)**

## Differential Operators

* Differential Operators - when applied to the image returns some derivatives.
* Model these "Operators" as masks/kernels that compute the image gradient function
* Threshold the this gradient function to select the edge pixels.

## Image Gradient

gradient points : [delta f / delta x, delta f / delta y]

* The gradient points in the direction of most `rapid` increase in intensity.

## Discrete Gradient

## Operators

* Sobel
* Prewitt
* Roberts

using `fspecial('sobel')` in matlab to create this operators.

`imfilter()` 默认使用 `correlation`

## Derivative theorem of convolution

* 卷积的积分 等于 先对高斯核积分，再卷积原图像
* 如何获得积分的极值：
  * 再来一次积分，（对高斯核进行两次积分，然后再卷积源图像）
  * 对高斯核两次积分，就是 Laplician of Gaussian

## 问题
如何解决 正梯度和负梯度
