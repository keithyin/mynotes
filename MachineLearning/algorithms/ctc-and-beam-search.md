# CTC 与 BeamSearch

**真实标签**

$Y=(y_1, y_2, ..., y_U)$ 这是一个样本的标签，对阵势标签处理一下，得到

$Z=(\varepsilon, y_1, \varepsilon, y_2, ..., \varepsilon, y_U, \varepsilon)$ .  



$\alpha_{s,t}$ : 时刻 `t` , 得到的序列 能够 `Merge` 成 `Y[:(s+1)/2]` 的概率。  `Y[:(s+1)/2]`就是 `Z[:s] `中包含的 `token`。

![https://distill.pub/2017/ctc/assets/ctc_cost.svg](https://distill.pub/2017/ctc/assets/ctc_cost.svg)



**Step**:

* 为什么用 CTC
* 为啥引入 $blank$ 标签
* CTC 的基本思路
* 对 GroundTruth 进行的处理
* 动态规划的 $\alpha$ 定义
* 二维图的解释



## BeamSearch

为什么 BeamSearch， 与 Viterbi decoder 的不同。

**Viterbi** : 概率最大的路径

* 总路径概率最大，那么 0～t 时刻的路径也是概率最大的。
* 假设已经得到 0~t-1 时刻到达各个状态的最优路径，那么如何获取 t 时刻到各个状态的最优路径
* 因为 t 时刻的任意状态可以由 t-1 时刻的任意状态转移得到，
* 所以当然是选 概率最大的那一个



**BeamSearch** : 估计的概率最大的路径

* 不考虑 t-1 时刻的所有状态，仅仅考虑前 K 个高的。然后向 t时刻的状态转移



## 参考资料

[https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)