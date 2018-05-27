**学习方法（和拥有的数据集有关）**

* 监督学习
* 非监督学习
* 半监督学习



**模型种类（和对数据建模的方式相关）**

* 生成模型
* 判别模型



# 生成模型

**对于生成模型我们通常对 $P(x; \theta)$ 或者 $P(x, z; \theta)$ 建模**



**目前的生成模型可以分为以下几类**

* GAN
* VAE
* autoregressive  model



**生成模型可以做什么呢？**

* density estimation, 给定一个 x 输出 $P(x)$
* 图片生成，输入噪声，输出逼真的图片 (GAN, VAE 都可以办到)
* 图片修改，修改图片的某些特征，（VAE 可以做到， GAN 应该也可以）
* 图片复原，给张有缺陷的图片，还原成原始图片（VAE可以做到，GAN应该也可以）



**density estimation**

----

* 我们有一堆无标签的数据，$D=\{\mathbf x^{(1)}, \mathbf x^{(2)}, ..., \mathbf x^{(N)}\}$ , 而我们关心的是 recovering or estimating 它们的概率密度 $p(x)$ 。
* [视频地址](https://vimeo.com/252105837)
* wavenet 和 pixel rnn 不仅仅能够从模型中采样图片/音频，同样也具备 density estimation 的能力。
* 两种 density estimator
  * autoregressive models
  * normalizing flows  ： **从一个简单分布中采样 $\mathbf u \sim N(0, I)$ , 然后经过 神经网络变换 的到一个数据点 $\mathbb x$ 。**结构就像是 GAN 中的 G 模型，但是有一点不同，神经网络表示的函数 必须是 invertible 的。这样的话，给定任意 $\mathbf x$ ，都可以估计处其概率 $p(\mathbf x) = p(\mathbf u) |\frac{\partial f}{\partial u}|^{-1}, \mathbf u = f^{-1}(\mathbf x)$ 。可以看出，由于 $f$ 可逆，所以可以通过 $ \mathbf u = f^{-1}(\mathbf x)$  计算得到 $\mathbf u$, 然后利用前一个公式就可以计算得到 $p(\mathbf x)$ .
* AR models **with Gaussian conditionals** are flows, 注意， with Gaussian conditionals 的才是 flow
  * ​