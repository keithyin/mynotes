# 图片超分辨率（GAN）



**intuition**

* 希望生成的 高分辨率 既有小的 MSE loss， 有能更好的 fool 判别器（生成的高分辩率图像在真实高分辩图排片的 manifold 上）





* adversarial loss （将生成的图片向 真实图片的 manifold 上推）
* content loss （内容 loss）



**GAN 的 判别器**

> Discriminator is trained to **differentiate** between the **super-resolved images** and **original photo-realistic images**.



**Content Loss**

> content loss motivated by **perceptual similarity** instead of similarity in pixel space.



最终目标是: 训练出一个 生成函数

* input :  LR input image
* output : corresponding HR counter part




## 缺点

* 判别器用了 全连接，不利于 向任何的 大小的图片扩展




## Glossary

* photo-realistic：逼真的图片
* perceptual quality：
* MOS (mean opinion score)

