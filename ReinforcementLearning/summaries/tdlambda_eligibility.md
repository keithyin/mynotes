# TD($\lambda$) 与 eligibility traces

**TD($\lambda$), the $\lambda$ refers to use of an eligibility trace.**



* eligibility trace 方法是介于 TD(0) 和 MC 方法之间的方法
* eligibility trace 方法提供了一个 使得 MC 方法可以用于 非 episode 问题的 实现



> Eligibility traces also provide a way of implementing Monte Carlo methods online and on continuing problems without episodes.



> eligibility traces offer an elegant algorithmic mechanism with significant computational advantages.



eligibility traces 的机制就是一个 `short-term memory vector` $\mathbb e_t \in \mathbb R^n$ 