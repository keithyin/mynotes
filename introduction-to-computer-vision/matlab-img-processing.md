
```c
h = fspecial('gaussian', fsize, sigma)
out = imfilter(im, h)
```

```c
noise_img = imnoise(img, "salt & pepper", 0.02);
midian_filtered = medfilt2(noise_img);
```

## show image

```c
imshow()
imshowpair(img1, img2, 'montage')

figure, surf(c), shading flat;
```
## img compution

```c
c = normxcorr2(img, kernel)
```
## 颜色map转换
```c
rgb2gray(img)
```

## utils func

```c
size(s,2) //返回s第二维的大小。
```

## 计算梯度
```c
img = double(imread(file_name)/255.0);
imshow(img);//imshow 默认认为double类型的范围为[0,1]

[gx, gy] = imgradientxy(img, 'sobel')

imshow((gx+4)/8); //将值 rescal 到 [0,1]

[gmag gdir] = imgradient(gx, gy);
imshow((gdir+180.0)/360.0);//angle in degrees[-180, 180]
```

## 计算梯度流程

1. 滤波：去噪声，噪声的导数很大
2. 然后归一化
3. 求梯度
