# opencv 直方图相关 API



## calHist

```python
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist
# images : list of images, image的 shape 可以是 二维或三维， 三维的最后一维是 channel，二维为1-channel
# channels: list of int 要对哪个 channel 求 histogram。如果是多个图片，可以考虑将 images concat 成一个 img，再考虑 channel。
# mask: 不想考虑的地方盖上
# histSize: list of int, 表示 返回 hist 的维度 即：多少 bin。
# ranges: list of int, 表示考虑什么范围内的 值
# 注意： len(channels)==len(histSize)== len(ranges)//2
```



```python
res = cv2.calcHist([rgb_img], channels=[0,1,2], 
             mask=None, histSize=[256,256,256], ranges=[0,256,0,256,0,256])

# res 是个 三维的值， 表示将 (b,g,r) 做为一个点的 统计直方图。
```









## 参考资料

[https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/histograms.html)

