# Normalizing Flow

**为什么需要 Normalizing Flow**

----





**总结一下**

------

-  Normalizing Flow 需要转换函数是 invertible的
  - 为啥 需要是 invertible 的？
  - invertible 好计算转换过去值的概率
  - 为啥要计算转换过去的值的概率？
- 希望 Normalizing Flow 能够扩展到高维空间，考虑到了 autoregressive flow
  - 为啥扩展到高维，需要 autoregressive
  - 猜测的可能原因：原始 Normalizing Flow 由于 bottleneck 层的影响，导高维建模能力堪忧，需要叠加好些层才能达到想要的效果。但是如果把 bottleneck 层换成 autoregressive nn，建模能力就大大增强了哟。
- 但是 autoregressive flow 的性质（这一步的值需要之前的值），所以不适合采样任务
- 所以出来另外一种流，inverse autoregressive flow，因为这种流的形式和 autoregressive flow 的形式互逆。这种流就好处多多了，采样贼快。
- ar 与 iar 只是两种不同的建模方式而已，不要纠结太多。



**normalizing flow, ar, iar 的 Jacobian 行列式都很好计算。（行列式好计算非常重要。）**

**为啥要行列式好计算？？？？？？？？？**