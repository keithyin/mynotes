# autoregressive models

# [Auto-Regressive Generative Models (PixelRNN, PixelCNN++) 博客笔记](https://towardsdatascience.com/auto-regressive-generative-models-pixelrnn-pixelcnn-32d192911173) 

**三类生成模型**

----

* GAN
* VAE
* autoregressive models



**GAN vs autoregressive models**

----

* **GAN** 学习一个分布映射函数 $f(\bullet)$，将 $x\_fake = f(z), z \sim N(0,1)$ ,    映射后的 $P(x\_fake)$ 和真实的图片分布一致。
  * 可以看出，使用 `GAN` 无法直接获得概率
  * 不过感觉改一下，就能得到图片的概率了，G 输出256个像素值的分类，然后采样(greedy or random)得到结果，再给到 D。这样就能知道 G 输出的图片的概率了。
* **autoregressive models** ：显式返回概率密度。
* **两个都在学习分布的映射**
* autoregressive 的训练比 gan 更加稳定
* autoregressive model 连续数据和离散数据都能搞，gan 搞离散数据有点难受？？？？？
* gan 可以产生高质量的图片，faster to train 。。。 faster to train? gan 不稳定。 tm 在逗我。



**autoregressive models**

----

* **modeling the distribution** of natural images , 即：建模 $P(x)$ ， 那么 $P(x)=?$ ，等于啥呢？
* $P(x)$ 需要 tractable and scalable
* ​



**pixel CNN ++**

----

* softmax layer **makes gradients sparse** early on during training. ????????



**discretized logistic mixture likelihood**

----

* 使用 softmax 的问题是，softmax 不知道 pixel value 128 和 pixel value 127/129 非常近。





# Normalizing Flow

**总结一下**

----

*  Normalizing Flow 需要转换函数是 invertible的
  * 为啥 需要是 invertible 的？
  * invertible 好计算转换过去值的概率
  * 为啥要计算转换过去的值的概率？
* 希望 Normalizing Flow 能够扩展到高维空间，考虑到了 autoregressive flow
  * 为啥扩展到高维，需要 autoregressive
  * 猜测的可能原因：原始 Normalizing Flow 由于 bottleneck 层的影响，导高维建模能力堪忧，需要叠加好些层才能达到想要的效果。但是如果把 bottleneck 层换成 autoregressive nn，建模能力就大大增强了哟。
* 但是 autoregressive flow 的性质（这一步的值需要之前的值），所以不适合采样任务
* 所以出来另外一种流，inverse autoregressive flow，因为这种流的形式和 autoregressive flow 的形式互逆。这种流就好处多多了，采样贼快。
* ar 与 iar 只是两种不同的建模方式而已，不要纠结太多。



**normalizing flow, ar, iar 的 Jacobian 行列式都很好计算。（行列式好计算非常重要。）**

**为啥要行列式好计算？？？？？？？？？**

