# Model Free Control

回忆 Planning 中 Policy Iteration 算法
* Policy Evaluation：根据当前policy采样一堆数据，然后进行 policy evaluation. (Monte-Carlo, TD ?)
* Policy Improvement：根据 policy evaluation 结果 进行 improve policy。Greedy Policy Improvement?
* 👆上面两个操作不停循环

Policy Iteration包含两个主要阶段：Policy Evaluation & Policy Improvement, 对于 Policy Evaluation，是有 modelf-free(MC, TD). 如果想要整体都Model Free的话，那么 Policy Improvement 也需要 Model Free方法。

# Policy Improvement

* Greedy Policy improvement over $V(S)$ requires model of MDP
$$
\pi'(s) = \arg \max_{a \in A} \mathcal R_s^a + \mathcal P_{ss'}^aV(s')
$$

* Greedy policy improvement over $Q(s, a)$ is model-free
$$
\pi'(s) = \arg \max_{a \in A} Q(s, a)
$$

因为 Greedy Policy Improvement over $Q(s, a)$ is model-free，所以在 policy evaluation的时候，我们 evaluate 的也是 $q(a,s)$ 而非 $v(s)$ 了。



#  on-policy
> * on-policy 与 off-policy 的区分出现在 policy-evaluation 阶段。
> * off-policy使用 behavior-policy 产生的 trajectory，来 evaluate target-policy。

基础的 Policy Iteration算法，在 policy evaluation时候，因为需要采样大量episode，旨在更精确的评估policy，所以需要耗费大量时间。对于control问题来说，policy evaluation阶段，我们需要耗费那么长时间吗？
* 答案当然是 可以不费那么长时间，是有一个 episode 进行 policy evaluation 即可，然后执行 policy improvement
* MC On-Policy

Monte-Carlo中我们需要一个 episode 进行 policy evaluation。对于TD来说，一个 time-step 我们就可以进行 policy evaluation 然后 policy improvement了。对应的算法也叫做 Sarsa
* TD On-Policy

# Off-Policy
> * on-policy 与 off-policy 的区分出现在 policy-evaluation 阶段。
> * off-policy使用 behavior-policy 产生的 trajectory，来 evaluate target-policy。

off-policy的优点
* Learn from observing humans or other agents
* Re-use experience generated from old policies $\pi_1, \pi_2, \pi_3, ..., \pi_{t-1}$
* Learn about optimal policy while following exploratory policy
* Learn about multiple policies while following one policy

在policy-evaluation的时候，我们的目标主要是计算 $q_\pi(a,s)$, $q_\pi(a,s)=\mathbb E_\pi[G_t|S_t=s, A_t=a]$. 如果我们使用其它 policy $\mu$ 采样出来的 trajectory 来进行policy evaluation的话，那就需要 importance sampling
![](https://img-blog.csdnimg.cn/2020123009222658.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI0MzYxNDk=,size_16,color_FFFFFF,t_70)
![](https://img-blog.csdnimg.cn/20201230092022439.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI0MzYxNDk=,size_10,color_FFFFFF,t_10)
![](https://img-blog.csdnimg.cn/20201230092101210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTI0MzYxNDk=,size_10,color_FFFFFF,t_10)

