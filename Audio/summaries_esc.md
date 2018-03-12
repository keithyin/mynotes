# ESC数据集论文总结



**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**

> we propose to use Convolutional Restricted Boltzmann Machine (ConvRBM) to learn filterbank from the raw audio signals.



> ConvRBM is a generative model trained in an unsupervised way to model the audio signals of arbitrary lengths. ConvRBM is trained using annealed dropout technique and parameters are optimized using Adam optimization.



> We have used our proposed model as a front-end for the Environmental Sound Classification (ESC) task with supervised Convolutional Neural Network (CNN) as a back-end. Using CNN classifier, the ConvRBM filterbank (ConvRBM-BANK) and its **score-level fusion** with the Mel filterbank energies (FBEs) gave an absolute improvement of 10.65 %, and 18.70 % in the classification accuracy, respectively, over FBEs alone on the ESC-50 database. This shows that the proposed ConvRBM filterbank also contains highly **complementary information** over the Mel filterbank, which is helpful in the ESC task.

* 用了 fusion 之后效果很好
* 只用ConvRBM-BANK 效果会如何?
* 提出的算法对 MFCC 特征有很好的补充作用.

**Audio Event and Scene Recognition: A Unified Approach using Strongly and Weakly Labeled Data**

> In this paper we propose a novel learning framework called **Supervised and Weakly Supervised Learning** where the goal is to learn simultaneously from weakly and strongly labeled data