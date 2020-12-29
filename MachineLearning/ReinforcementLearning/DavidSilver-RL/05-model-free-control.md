# Model Free Control

回忆 Planning 中 Policy Iteration 算法
* Policy Evaluation：根据当前policy采样一堆数据，然后进行 policy evaluation. (Monte-Carlo, TD ?)
* Policy Improvement：根据 policy evaluation 结果 进行 improve policy。Greedy Policy Improvement?
* 👆上面两个操作不停循环

# Policy Improvement

* Greedy Policy improvement over $V(S)$ requires model of MDP
$$
\pi'(s) = \arg \max_{a \in A} \mathcal R_s^a + \mathcal P_{ss'}^aV(s')
$$

* Greedy policy improvement over $Q(s, a)$ is model-free
$$
\pi'(s) = \arg \max_{a \in A} Q(s, a)
$$
