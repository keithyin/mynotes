# TRPO


$$
\eta(\pi) = \mathbb E_{s_0\sim\rho_0,a_t\sim\pi(\star|s_t)}\Biggr[\sum_{t=0}^\inf \gamma^tr_t\Biggr] \tag1
$$



$$
\eta(\pi) = \eta(\pi_{old}) + \mathbb E_{\tau\sim\pi}\Biggr[\sum_{t=0}^\inf \gamma^tA^{\pi_{old}}(s_t,a_t)\Biggr] \tag2
$$

$$
A^{\pi_{old}}(s,a) = \mathbb E_{s'\sim P(s'|s,a)} \Biggr[r(s,a)+\gamma V^{\pi_{old}}(s')-V^{\pi_{old}}(s)\Biggr] \tag3
$$



**Advantage**

* 当前状态下选择某个 action 获得回报，与 期望回报的 差值

**如何理解第二个式子：**

* 如何判断 $\pi$ 是否比 $\pi_{old}$ 好呢？
* 从 $\pi$  中采 trajectory，然后放在 $\pi_{old}$ 的值函数 中 求 Advantage
* 只要 采的 trajectory 的 action 选择 优于，期望值，说明新的 policy 还不错
* 这里要求只是高于 期望值就好，并不是优于 $\pi_{old}$


$$
\nabla_{\theta} L(\pi_{\theta})|_{\theta_{old}} = \mathbb E_{s,a\sim\pi_{old}}\Biggr[\nabla_\theta\log\pi_{\theta}(a|s)A^{\pi_{old}}(s,a)\Biggr]
$$
