# Normalizing Flow

**什么是 Normalizing Flow**

----

* a **simple initial density** is transformed into a **more complex one** by applying a **sequence of invertible transformations** until a desired level of complexity is attained.
  * 有个疑问：为什么要是 **invertible transformations ???????????????????**
  * 因为 invertible 的话，给定 $z \in p(z), y = f(z)$, 就能算出 $p(y)$
  * 如果想要计算 $p(y)$ 轻松一些的话， 行列式的值和 $f^{-1}$ 也需要 trivial 的。




* invertible transformations of densities can be used to implement more complex densities。
  * 随机变量函数的分布，函数是 invertible 的。$Y=g(X)$ 
* $Y=g_3(g_2(g_1(X)))$ , 这个就是 Normalizing Flow。将 transformation chain 起来。





**Normalizing Flow有啥作用**

----

* 从简单分布（正态分布， etc）中采样得到 $\mathbf z$ , 经过函数变化得到 $\mathbf y$ ,  得到的$P(\mathbf y)$ 是非常复杂的。
* 即：将 **简单分布** 中采样的样本变换成 **复杂分布**采样的样本
* 缓解了 VAE 的 mean-field assumption 的问题。







**为什么需要 Normalizing Flow**

----

* VAE 中，RL 中。用来干什么？






**Autoregressive Models are Normalizing Flows**

----

* ​






**总结一下**

------

-  Normalizing Flow 需要转换函数是 invertible的
  - 为啥 需要是 invertible 的？
  - invertible 好计算转换过去值的概率
  - 为啥要计算转换过去的值的概率？
-  希望 Normalizing Flow 能够扩展到高维空间，考虑到了 autoregressive flow
  - 为啥扩展到高维，需要 autoregressive
  - 猜测的可能原因：原始 Normalizing Flow 由于 bottleneck 层的影响，导高维建模能力堪忧，需要叠加好些层才能达到想要的效果。但是如果把 bottleneck 层换成 autoregressive nn，建模能力就大大增强了哟。
-  但是 autoregressive flow 的性质（这一步的值需要之前的值），所以不适合采样任务
-  所以出来另外一种流，inverse autoregressive flow，因为这种流的形式和 autoregressive flow 的形式互逆。这种流就好处多多了，采样贼快。
-  ar 与 iar 只是两种不同的建模方式而已，不要纠结太多。



**normalizing flow, ar, iar 的 Jacobian 行列式都很好计算。（行列式好计算非常重要。）**

**为啥要行列式好计算？？？？？？？？？**





## Normalizing Flows and Variational auto-encoder





## NICE: Non-linear Independent Components Estimation

> Learning a highly non-linear transformation that maps the **training data** to space where its **distribution is approximately factorized**.  
>
> **By directly maximizing log-likelihood**.






## 参考资料

[http://akosiorek.github.io/ml/2018/04/03/norm_flows.html](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)

[https://blog.evjang.com/2018/01/nf1.html](https://blog.evjang.com/2018/01/nf1.html)

[https://blog.evjang.com/2018/01/nf2.html](https://blog.evjang.com/2018/01/nf2.html)

[TensorFlow Distributions](https://arxiv.org/pdf/1711.10604.pdf)



## 关于 Normalizing Flow 的论文

[Masked Autoregressive Flow for Density Estimation](https://arxiv.org/pdf/1705.07057.pdf)

