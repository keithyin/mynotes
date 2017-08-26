# 如何理解 TD($\lambda$) backward view

首先看 **Eligibility Traces** :  eligibility traces 是为了解决 credit assignment 问题，即：*到底应该把 credit 给 之前经历过的哪个状态。*

现在就面临两种选择：

* 把 credit 给 经常经历的 状态
* 把 credit 给 最近的状态

**Eligibility Traces** 就把这两种方式结合到了一起，所以得到下面的公式：
$$
\begin{aligned}
E_{0}(s) &= 0 \\
E_{t}(s) &= \gamma\lambda E_{t-1}(s) + \mathbb I(S_t=s)
\end{aligned}
$$
$E_0(s)$ 是初始状态，刚一开始初始化用的 

$E_t(s)$ 就表示：**假设我们一个 trace，时刻 为 t\_begin -> t\_end，那么我们，在 t\_end 时刻我们收获到了一个 credit，那么这个 credit 应该怎么 assignment 给在 t\_begin-> t\_end 之间经历过的状态（t\_end 不考虑，因为是终止状态）。$E_t(s)$ 就表示了 credit 分配的 权重。**



## backward view TD(0)

TD(0) : *take one step look ahead*，然后再回来更新当前状态。
$$
V(s) \leftarrow V(s) + \alpha\Bigr(R_{t+1}+\gamma V(S_{t+1})-V(S_t)\Bigr)
$$
我们将 $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$ 看作 **credit** ，如何将 这个 credit 分配给 当前状态呢？value function 的更新公式就变为：
$$
V(s) \leftarrow V(s) + \alpha\delta_tE_t(s)
$$
由于 我们这个 trace 只有 $s_t, s_{t+1}$ ，所以：
$$
\begin{aligned}
E_t(s) &= \mathbb I[S_t=s] \\
V(s) & \leftarrow V(s) + \alpha\delta_tE_t(s)\\
V(s) & \leftarrow V(s) + \alpha\delta_t
\end{aligned}
$$
跟原始的 TD(0) 更新公式一样。



