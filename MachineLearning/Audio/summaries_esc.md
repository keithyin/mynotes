# ESC数据集论文总结



**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**



> we propose to use Convolutional Restricted Boltzmann Machine (ConvRBM) to learn filterbank from the raw audio signals.



> ConvRBM is a generative model trained in an unsupervised way to model the audio signals of arbitrary lengths. ConvRBM is trained using annealed dropout technique and parameters are optimized using Adam optimization.



> We have used our proposed model as a front-end for the Environmental Sound Classification (ESC) task with supervised Convolutional Neural Network (CNN) as a back-end. Using CNN classifier, the ConvRBM filterbank (ConvRBM-BANK) and its **score-level fusion** with the Mel filterbank energies (FBEs) gave an absolute improvement of 10.65 %, and 18.70 % in the classification accuracy, respectively, over FBEs alone on the ESC-50 database. This shows that the proposed ConvRBM filterbank also contains highly **complementary information** over the Mel filterbank, which is helpful in the ESC task.

* 用了 fusion 之后效果很好
* 只用ConvRBM-BANK 效果会如何?
* 提出的算法对 MFCC 特征有很好的补充作用.



**Score-Level Fusion指的是什么?**

总结:

> 使用 ConvRBM 无监督训练出来的特征与 MFCC/Mel 特征进行融合, 然后会得到非常好的分类效果, 无监督训练出来的特征起到了非常好的特征补充作用.



**LEARNING FROM BETWEEN CLASS EXAMPLES FOR Deep SOUND RECOGNITION**

* 数据增强的一种方式
* soft-label 的感觉
* 数据mix可能出现的问题
  * 声音的大小决定了输出的概率?
* 两个声音混合变成了另一种声音?
  * ​

> Our strategy is to learn a discriminative feature space by **recognizing the between-class sounds as between-class sounds**.  We generate between-class sounds by mixing two sounds belonging to different classes with a random ratio. We then input the mixed sound to the model and train the model to output the mixing ratio.

* **Fisher’s criterion**
  * ​



**ENVIRONMENTAL SOUND CLASSIFICATION WITH CNN**





**Audio Event and Scene Recognition: A Unified Approach using Strongly and Weakly Labeled Data**

> In this paper we propose a novel learning framework called **Supervised and Weakly Supervised Learning** where the goal is to learn simultaneously from weakly and strongly labeled data