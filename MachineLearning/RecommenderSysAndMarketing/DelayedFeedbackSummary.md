# 生存分析💀💀

> https://zh.wikipedia.org/wiki/%E7%94%9F%E5%AD%98%E5%88%86%E6%9E%90
>
> https://www.statisticshowto.com/hazard-function/



基本符号

生存函数：$S(t) = Pr(T>t)$, T为死亡时间（随机变量）。$S(t)$ 表示：$t$ 时刻 **还** 活着的概率，CDF

$F(t) = 1-S(t) = Pr(T<=t)$：$t$ 时刻 **已经** 死亡的概率，是个 CDF。注意这个已经，只要是在 $t$ 之前死亡💀都包含

$f(t)=\frac{F(t)}{dt}$ : $t$ 时刻死亡的概率

Hazard Functin（危险函数）：$h(t)=\frac{f(t)}{S(t)}$, 是个条件概率，$t$  时刻如果还活着，那么在$t$ 时刻死的概率是多少。。这解释好怪异

* 该值越大，说明越危险，马上就要 game over了。

  





# Modeling Delayed Feedback in Display Advertising

> 2014

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.650.6087&rep=rep1&type=pdf

* 使用两个模型，一个预估转化率，一个预估转化时间。这样就将转化 和 时延 disentangle 了。



如何训练：

* 每个timeslot的训练都需要模型重新开始训练，不能热启动
* 每个timeslot训练都需要回溯N个timeslot之前的样本，根据当前时间为其打标签，然后扔进模型训练



# A Nonparametric Delayed Feedback Model for Conversion Rate Prediction

> 2018

https://arxiv.org/pdf/1802.00255.pdf

