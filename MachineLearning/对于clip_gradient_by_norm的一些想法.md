# 有 clip_grad_norm 引发的思考

在训练RNN模型的时候大多数时候我们都会使用 `clip_grad_norm` 来对求得的参数梯度求 `norm`, 但是 `max_norm` 的值应该设置成多少才合适呢?



**先看一下 一个 tensor 的 norm 值为多少**

```python
import torch
from torch.nn import init
import math
a = torch.FloatTensor(300, 200)
init.xavier_normal(a)
print(math.sqrt(math.sqrt(300 * 200))) # 15.650845800732874
print(torch.norm(a))                   # 15.448935957669539

a = torch.FloatTensor(300, 200)
init.normal(a)
print(math.sqrt(300 * 200)) # 244.94897427831782
print(torch.norm(a))        # 244.56821672888876 可以看出, l2norm 其实就是计算 mean=0的标准差
```



从公式和上面的结论可以看出, `l2norm` 其实就是在计算 `mean=0` 的标准差, `clip_grad_norm` 意味着我们希望梯度的分布是什么样子的. 



假设有 150 个参数, `max_norm=5` 其实意味着, 
$$
\sqrt {x_1^2+x_2^2+x_3^2+x_4^2+...+x_{150}^2} \le 5
$$
每个参数的方差为 $\frac{{max\_norm}^2}{{num\_params}}=\frac{25}{150}$

