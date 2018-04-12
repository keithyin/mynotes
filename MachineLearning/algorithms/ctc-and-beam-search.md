# CTC 与 BeamSearch

**真实标签**

$Y=(y_1, y_2, ..., y_U)$ 这是一个样本的标签，对阵势标签处理一下，得到

$Z=(\varepsilon, y_1, \varepsilon, y_2, ..., \varepsilon, y_U, \varepsilon)$ .  



$\alpha_{s,t}$ : 时刻 `t` , 得到的序列 能够 `reduce` 成 `Y[:(s+1)/2]` 的概率。  `Y[:(s+1)/2]`就是 `Z[:s] `中包含的 `token`。

![https://distill.pub/2017/ctc/assets/ctc_cost.svg](https://distill.pub/2017/ctc/assets/ctc_cost.svg)







## 参考资料

[https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)