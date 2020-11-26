# Markov

## Markov Process（马尔科夫过程）

**马尔科夫性质：**

The future is independent of the past given the present.(给定当前的状态，未来就和过去无关)

数学表达式就是 $\text{P}[S_{t+1}|S_t]=\text{P}[S_{t+1}|S_t,S_{t-1},...]$

* 当前状态捕捉到了所有相关的历史信息
* 一旦当前的状态已知，那么历史消息就可以扔掉了
* The state is a sufficient statistic of the future

马尔科夫过程是一个无记忆的随机过程，即：一个随机状态序列 $S_1,S_2,....$满足马尔科夫性质

**一个马尔科夫过程可以表示成为一个二元组<$\mathcal{S},\mathcal{P}$>**

* $\mathcal{S}$ 是一个有限状态集合
* $\mathcal{P}$ 是状态之间的转移概率 $\mathcal P_{ss'}=\mathbb P(S_{t+1}=s'|S_t=s)$



## Markov Reward Process(马尔科夫奖励过程)

**A Markov reward process is a Markov chain with values.**

* 首先是一个`Markov Process`
* 然后是离开某个状态都有一个奖励值，（注意奖励，是在离开状态的时候获得的）



一个 **Markov Reward Process** 是一个四元组 <$\mathcal{S},\mathcal{P},\mathcal{R},\gamma$>

* $\mathcal{S}$: 有限状态的集合
* $\mathcal{P}$: 状态转移矩阵 $\mathcal P_{ss'}=\mathbb P[S_{t+1}=s'|S_t=s]$
* $\mathcal{R}$:  奖励函数 $\mathcal R_s=\mathbb E[\text{R}_{t+1}|S_t=s]$,
  * 为什么下标是$t+1$呢？，这里强行解释一下，因为$Reward$是在离开$s$状态时获得的，所以是$\text{R}_{t+1}$
* $\gamma$:  衰减因子



**关于`Return`的定义：**

The return $G_t$ is the **total discounted reward from time-step $t$**


$$
G_t=R_{t+1}+\gamma R_{t+2} ... = \sum_{k=0}^{\inf}\gamma^k R_{t+k+1}
$$

$R_t$是个随机变量，所以$G_t$也是一个随机变量



**关于value function（值函数）的定义：**

The state value function $v(s)$ of an MRP is the **expected return** starting from state $s$


$$
v(s)=\mathbb E[G_t|S_t=s]
$$
**Bellman Equation for MRPs**:

值函数可以被分解成两个部分：

* immediate reward $\text{R}_{t+1}$
* discounted value of successor state $\gamma v(S_{t+1})$


$$
\begin{aligned}
v(s)&=\mathbb E[G_t|S_t=s] \\
&=\mathbb E[R_{t+1}+\gamma R_{t+2} + \gamma^2R_{t+3}+...|S_t=s]\\\\
&=\mathbb E[R_{t+1}+\gamma (R_{t+2} + \gamma R_{t+3}+...)|S_t=s] \\\\
&=\mathbb E[R_{t+1}+\gamma G_{t+1}|S_t=s]\\\\
&=\mathbb E[R_{t+1}+\gamma v(S_{t+1})|S_t=s]\\\\
v(s)&=\mathcal R_s+\gamma\sum_{s'\in S}\mathcal P_{ss'}v(s')
\end{aligned}
$$

**对于MRP来说，我们的目标就是求$v(s)$**



## Markov Decision Process(马尔科夫决策过程)

**A Markov decision process(MDP) is a Markov reward process with decisions.**(MDP比MRP多了一个决策项)



**MDP是一个五元组<$\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma$>**

* $\mathcal{S}$是一个有限状态集合
* $\mathcal{A}$是一个有限动作集合(actions)
* $\mathcal{P}$ 依旧是状态转移矩阵，只不过公式有点小变化，
  * $\mathcal P_{ss'}^a=\mathbb P[S_{t+1}=s'|S_t=s,A_t=a]$
* $\mathcal{R}$是奖励函数
  * $\mathcal R_s^a= \mathbb E(R_{t+1}|S_t=s,A_t=a)$
* $\gamma$ 是衰减因子



对于`MDP` 还有一个要考虑的就是`policy` $\pi$ ,它的定义如下：

* $\pi(a|s)=\Bbb P[A_t=a|S_t=s]$
* 意思是，在状态$s$下，应该采取什么样的动作 `action`
* **注意**: `policy` 既可以是`deterministic`的，也可以是`stochastic`的：
  * `deterministic`: $a=\pi(s)$
  * `stochastic`: $\pi(a|s)=\Bbb P[A_t=a|S_t=s]$




**给定MDP和$\pi$, MDP是可以转化为MRP的：**

* <$\mathcal S,\mathcal A,\mathcal P,\mathcal R,\gamma$>($\pi$) --> <$\mathcal S,\mathcal P^\pi,\mathcal R^\pi,\gamma$>
* $\mathcal P_{ss'}^\pi=\sum_{a\in A}\pi(a|s)\mathcal P_{ss'}^a$
* $\mathcal R^\pi=\sum_{a\in A}\pi(a|s)\mathcal R_s^a$




**MDP的两类值函数(value function):**

* state-value function $v_\pi(s)$
  * $v_\pi(s)=\Bbb E[G_t|S_t=s]$
  * 当前状态能够获得的期望`return`
* action-value function $q_\pi(s,a)$
  * $q_\pi(s,a)=\Bbb E_\pi[G_t|S_t=s, A_t=a]$
  * 当前状态下，选择`Action` `a`，获得的期望`return`
* 两类值函数之间的关系 (下面公式也叫做 Bellman Expectation Equation)(给定$\pi$，可以用来求$v_\pi(s), q_\pi(s,a)$)
  * $v_\pi(s)=\sum_{a\in A}\pi(a|s)q_\pi(s,a)$

  * $q_\pi(s,a)=\mathcal R_s^a+\gamma\sum_{s'\in S}\mathcal P_{ss'}^av_\pi(s')$
  * $v_\pi(s)=\sum_{a\in A}\pi(a|s)(\mathcal R_s^a+\gamma\sum_{s'\in S}\mathcal P_{ss'}^av_\pi(s'))$
  * $q_\pi(s,a)=\mathcal R_s^a+\gamma\sum_{s'\in S}\mathcal P_{ss'}^a\sum_{a'\in A}\pi(a'|s')q_\pi(s',a'))$




**最优值函数：**

最优值函数的定义只出现在`MDP`这里，因为有`D`嘛，所以就可以找到`D`，使得值函数最优：

* 最优 state-value function $v_\star(s)$

  * $v_\star(s)=max_\pi v_\pi(s)$

  * 即，找到最优的$\pi$ 使得 $v_\pi(s)$的值最大，这个最大的$v_\pi(s)$就是我们想要的
* 最优action-value function $q_\star(s,a)$
  * $q_\star(s,a)=max_\pi q_\pi(s,a)$
  * 即，找到最优的$\pi$ 使得 $q_\pi(s,a)$的值最大，这个最大的$q_\pi(s,a)$就是我们想要的



**Bellman Optimality Equation**

$$
\begin{aligned}
v_\star(s) &= \max_a q_\star(s,a) \\\\
 q_\star(s,a)&= \mathcal R_s^a+\gamma\sum_{s' \in S}\mathcal P_{ss'}^av_\star(s') \\\\
 v_\star(s) &= \max_a \Bigl(  \mathcal R_s^a+\gamma\sum_{s' \in S}\mathcal P_{ss'}^av_\star(s') \Bigl)\\\\
 q_\star(s,a)&= \mathcal R_s^a+\gamma\sum_{s' \in S}\mathcal P_{ss'}^a  \max_{a'} q_\star(s',a')
\end{aligned}
$$

* 以上式子可以用来求解最优 `policy`
