* 生存分析是啥
  *  analyzing the expected duration of time until one event occurs。例如，生物学中的死亡，或者机器的失败。
* 回答什么问题
  * what is the proportion of a population which will survive past a certain time? Of those that survive, at what rate will they die or fail? Can multiple causes of death or failure be taken into account? How do particular circumstances or characteristics increase or decrease the probability of survival?



模型：

* survial function：$S(t)$  is the probability that a subject survives longer than time *t*. (活的时间比 $t$ 长的概率 , $t$ 时刻还活着的概率)
  * $S(t)=Pr(T>t)$
* Lifetime distribution function: $F(t)$ ,活不到 $t$ 的概率
  * $F(t)=Pt(T\le t) = 1-S(t)$
* Lifetime distribution density function. **某个时刻死亡** 的概率密度
  * $f(t)=F'(t)=\frac{d}{dt}F(t)$
  * $s(t)=S'(t)=-f(t)$

* hazard function: $\lambda(t)=\frac{f(t)}{S(t)}=-\frac{S'(t)}{S(t)}$





https://en.wikipedia.org/wiki/Survival_analysis

https://www.statisticshowto.com/hazard-function/
