# Dueling Network Architecture For DRL

**key contribution**

* 提出了一个新的 网络框架 （Dueling Network）
* lead to better policy evaluation in the presence of many similar-valued actions.
* outperform the state-of-art on the Atari 2600 domain
* New Architecture, older algorithms



**Dueling Network**

* two separate estimators
  * one for state value function
  * one for state-dependent action **advantage** function



> The dueling network has two streams to separately estimate **state-value** and the **advantages** for each action.

为什么说 预测了  advantages 了呢？ 从什么方面体现出来的。



> The two streams are combined via a special aggregating layer to produce an estimate of the state-action value function $Q$ .

真有意思。。。。



> The dueling network automatically produces separate estimates of the state value function and advantage function, without any extra supervision.

给出的这个结论文章中有论证吗？？？？



> However, we need to keep in mind that $Q(s,a;\theta,\alpha,\beta)$  is only a parameterized estimate of the true Q-function. Moreover, it would be wrong to conclude that $V(s;\theta,\beta)$ is a good estimator of the state-value function.or likewise $A(s,a;\theta,\alpha)$ provides a reasonable estimate of the advantage function.

实力打自己脸嘛，上句还说 `estimates of the state value function and advantage function` ，这就虚了？？？？ 不是一个 好的 estimate， 那还说是 estimate 啊，无语。 这不就相当于，我有一个 猫狗分类器，但是正确率就 40, 谁TM 这分类器是干嘛的。



**关于如何结合所谓的 state-value 和 advantages**

第一个方法是这样：
$$
Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + A(s,a;\theta,\alpha)
$$
这个是 Advantage 的定义式，但是这个式子不适合做最后一层，因为：

* given $Q$ , 无法唯一的还原 $V$ 和 $A$ ，因为 $V,A$ 一个加常量，一个减常量，得到的 $Q$ 还是一样



第二个方法：
$$
Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Biggr(A(s,a;\theta,\alpha)-\max_{a'\in \mathcal A}A(s,a';\theta,\alpha)\Biggr)
$$
然后就会有后续公式
$$
\begin{aligned}
a^* &= \arg\max_{a'\in \mathcal A}Q(s,a';\theta,\alpha,\beta)\\
&= \arg\max_{a'\in \mathcal A}A(s,a';\theta,\alpha)
\end{aligned}
$$
然后就会得到：
$$
Q(s,a^*;\theta,\alpha,\beta)  = V(s;\theta,\beta)
$$
这样，$V(s;\theta,\beta)$ 就是用来估计 state-value function 的，其他部分就是用来 估计 advantages 的



第三个方法：
$$
Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + \Biggr(A(s,a;\theta,\alpha)-\mathbb E_{a'\in \mathcal A}A(s,a';\theta,\alpha)\Biggr)
$$

* force advantage function estimator to have zero advantage at the chosen action





## 强行理解一波

直接学习 q-value 变成了，学习了 q-value 的组合，组合计还是比较简单的。





## Ideas

* 能否用在 continuous action space 上
* 用在 policy gradient 那一套方法上
* 在网络内 将问题 decompose 是个有意思的事情呢？ 还有什么任务可以在网络内 decompose 的？？？

