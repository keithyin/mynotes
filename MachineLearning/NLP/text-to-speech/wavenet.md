# wave net

**解决了什么问题**

* 使用神经网络进行**原始波形生成**

**怎么解决的**

* 使用了一个 fully probabilistic and autoregressive 的模型

**为什么能解决**

* 玄学



**使用自回归神经网络训练波形生成器/音频合成器/vocoder** 



**核心组成部分**

* 自回归
* dilated causal convolution (也就是 kernel size 为 2 的 dilated convolution)
* 对比做 **lstm** 训练的语言模型，惊人的相似，除了 **lstm** 无法解决序列过长的问题。



**思考**

* 为什么不用 **LSTM** ，却使用 **dilated casual convolution**
  * 语音信号很长，**LSTM** 无法捕捉太长的时序信息。
  * **LSTM** 的时序特性 导致训练的时候非常慢



**参考资料**

[https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

[https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/](https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/)

[https://github.com/oxford-cs-deepnlp-2017/lectures](https://github.com/oxford-cs-deepnlp-2017/lectures)



# Parallel wave-net : Fast High-Fidelity Speech Synthesis





# 所需知识

**autoregressive model** (自回归模型)

* [https://baike.baidu.com/item/%E8%87%AA%E5%9B%9E%E5%BD%92/12732327](https://baike.baidu.com/item/%E8%87%AA%E5%9B%9E%E5%BD%92/12732327)
* 即：$x_t$ 的预测值，与 $x_{1\rightarrow (t-1)}$ 的预测之相关。



**inverse autoregressive flows**

* ​