## TTS (Text to Speech)

**什么是 TTS 呢？ 即： 输入是文本,输出是声音波形， 这个任务和语音识别互为反任务。**



**TTS 可以大概分成两个步骤：**

* 从文本中抽取足量信息
* 想进办法生成波形


**建模的时候分成三个部分：**

> A text-to-speech synthesis system typically consists of multiple stages, such as:
>
> * a text analysis
> * a acoustic model
> * an audio synthesis
>
> 摘自 Tacotron： Towards End-to-End Speech Synthesis，
>
> 如果想要对这三个部分有更具体的了解，可以读一下，Deep Voice：Real Time Neural Text-to-Speech

* text analysis fronted : 文本分析前端
  * 抽取音素特征，重音。等等语音特征。
* acoustic model ： 给定音素，输出声学特征（`duration prediction` , `fundamental frequency` 等等）。
* audio synthesis module ： 根据声学特征，合成声音









## Glossary

* **vocoder** [https://whatis.techtarget.com/definition/vocoder](https://whatis.techtarget.com/definition/vocoder)
  * A vocoder is an audio processor that captures the characteristic elements of an an audio signal and then uses this characteristic signal to affect other audio signals.  即：用某段音频的特征来影响另一个音频的特征。
  * 功能：提取源声音特征，将特征用到目标声音上。进行声音合成
  * 深度学习中提到的 **vocoder** 简单看作 **audio synthesis model** 就好。
* **phoneme** ：音素，音位
  * 语音中最小的单位，依据音节里的发音动作来分析，一个动作构成一个音素，音素又分为 **元音、辅音** 两大类
* **grapheme** ： 字形，字素，字母
  * 也就是 **word** , 单词。**hello world** 包含两个 `grapheme` 。
* **linguistic**
  * 语言学
* **fundamental frequency** : [https://en.wikipedia.org/wiki/Voice_frequency](https://en.wikipedia.org/wiki/Voice_frequency)
  * ​




**语言学一些东西**

* **Vowels** ：元音/母音
* **Consonants** : 辅音
* **grapheme** ： 表示单词
* **phoneme** : 表示音素，音素又分成元音，辅音两大类。
  * 英文的音素看作音标就可以了
  * 中文的音素
* **stress** ： 表示重音。



## 论文资料

**DeepVoice**

* [Deep Voice: Real-time Neural Text-to-Speech](http://cn.arxiv.org/pdf/1702.07825.pdf)

**Wavenet**

* [WaveNet: A generative model for raw audios](http://cn.arxiv.org/pdf/1609.03499.pdf)
* [FAST WAVENET GENERATION ALGORITHM](http://cn.arxiv.org/pdf/1611.09482.pdf)
* [Parallel WaveNet: Fast High-Fidelity Speed Synthesis](http://cn.arxiv.org/pdf/1711.10433.pdf)
* ​

**Tacotron**

* [Tacotron: Towards End-to-End Speech Synthesis](http://cn.arxiv.org/pdf/1703.10135.pdf)
* [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](http://cn.arxiv.org/pdf/1803.09047.pdf)




**tts 用到的技术**

* [autoregressive & inverse autoregressive]()

## 参考资料

[https://www.zhihu.com/question/26815523](https://www.zhihu.com/question/26815523)

[https://zhuanlan.zhihu.com/p/36737737](https://zhuanlan.zhihu.com/p/36737737)

[https://www.zhihu.com/question/50509644](https://www.zhihu.com/question/50509644)

[https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/](https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/)

[http://www.cstr.ed.ac.uk/downloads/publications/2010/king_hmm_tutorial.pdf](http://www.cstr.ed.ac.uk/downloads/publications/2010/king_hmm_tutorial.pdf)

[https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

[https://google.github.io/tacotron/](https://google.github.io/tacotron/)