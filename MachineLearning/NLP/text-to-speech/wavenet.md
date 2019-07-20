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



**wavenet 如何解决 TTS 问题**

* 将文本特征作为条件输入：linguistic features, predicted log fundamental frequency(F0), phoneme durations
* 然后训练 wavenet 网络。
* 文本特征可以使用原始方法提取，也可以使用神经网络提取。



**参考资料**

[https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

[https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/](https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/)

[https://github.com/oxford-cs-deepnlp-2017/lectures](https://github.com/oxford-cs-deepnlp-2017/lectures)



# Parallel wave-net : Fast High-Fidelity Speech Synthesis

**使用了 inverse autoregressive flow，这样 inference 的时候就可以并行了，速度飕飕的。**



# 所需知识

**autoregressive model** (自回归模型)

* [https://baike.baidu.com/item/%E8%87%AA%E5%9B%9E%E5%BD%92/12732327](https://baike.baidu.com/item/%E8%87%AA%E5%9B%9E%E5%BD%92/12732327)
* 即：$x_t$ 的预测值，与 $x_{1\rightarrow (t-1)}$ 的预测之相关。



**inverse autoregressive flows**

* [norm_flows](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)
* [masked-autoencoders](http://www.inference.vc/masked-autoencoders-icml-paper-highlight/)
* [Normalizing Flows Tutorial](https://blog.evjang.com/2018/01/nf1.html)






# 数据预处理与实现细节

**实现细节**

* 如何 `casual convolution` ：

  * 计算到达 `same` 效果需要多少 `padding`，然后全部 `pad` 到输入的左边，然后计算卷积。
  * 或者使用原始 same 卷积，但是 weight 的部分进行置 0 处理。


  * 看到很多实现在使用 valid 卷积，不明觉厉

**数据预处理**

* `waveform` ： rescale 到 [-1, 1] 区间内
* `mel_db` ： rescale 到 [0, 1] 区间内
* **对 waveform 进行 padding** (注意这个不是 mini-batch 时的 padding)，保证 `padded_waveform % hop_length == 0 , len(padded_wf)/len(mel) == hop_length` 
  * 用什么值进行 pad 呢？使用静音的值还是使用 reflect pad 方式？



* `mini-batch` 训练的时候，也需要对数据集进行 `padding` 。 input 记得开始用 `GO` 表示。然后预测第一个时刻的 输出值。
  * 其实也就是 输入的 shape 为 `[1,1,1]`, 然后使用 causal same pad 进行 padding，然后计算。

