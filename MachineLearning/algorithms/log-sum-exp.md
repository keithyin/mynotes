# LogSumExp

[from wikipedia](https://en.wikipedia.org/wiki/LogSumExp)

`LogSumExp` 函数是 `maximum` 函数的一个平滑近似。其定义是：
$$
LSE(x_1, x_2, x_3, ..., x_n) = \log\Bigr(\exp(x_1)+\exp(x_2) + ...+\exp(x_n)\Bigr)
$$
定义域： $\mathbf x \in \mathbb R^n$

值域：$\mathbb R$

不等式关系：
$$
\max\{x_1, x_2, ..., x_n\} \le LSE(x_1, x_2, ..., x_n) \le \max\{x_1, x_2, ..., x_n\} + \log(n)
$$
在使用计算机时候经常使用一个 `trick` 为了避免数值计算错误


$$
LSE(x_1, x_2, x_3, ..., x_n) =x^*  + \log\Bigr(\exp(x_1-x^* )+\exp(x_2-x^* ) + ...+\exp(x_n-x^*)\Bigr)
$$


$$
x^* = \max \{x1, x2, ..., x_n\}
$$

* 避免了 $x$ 的值过大，导致的 `exp` 数值计算不稳定的问题
* $x$ 是 `log` 空间的值。
* 如果是用此来计算联合概率的话，$x$ 是 `log prob`




**可以用来做什么**

* 计算最大值
* 可以用于 log 空间的计算，log 到 exp 然后再到 log 空间。
* **概率的 `log` 空间的加法运算** 。


