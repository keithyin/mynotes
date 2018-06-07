# 梯度与传统CV
梯度这个概念在传统CV的上用的很多。这个梯度和我们平时说的函数的梯度是一个意思，但是在计算的时候略有不同，图像中的梯度多是使用算子来计算的。

## 梯度与edge的关系

* edges correspond to extrema of derivatives

```python
  0   0    0
-1/2  0  -1/2
  0   0    0

#左边的梯度和右边梯度的平均
```

## Sobel Operator

计算 pixel 的 梯度。

theta = atan2(gy, gx)

```python
       -1   0   1
1/8 *  -2   0   2
       -1   0   1



       1    2    1
1/8 *  0    0    0
      -1   -2   -1
```

## Prewitt

```python
-1  0  1
-1  0  1
-1  0  1

 1   1   1
 0   0   0
-1  -1  -1
```

## Roberts

```python
 0  1
-1  0

1   0
0  -1
```

## But in the real world

* 噪声问题，会导致梯度上下抖动的厉害
  * 解决方法： smooth image
  
