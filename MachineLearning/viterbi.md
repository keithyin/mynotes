# 维特比算法

**统计学习方法 (李航)** 维特比算法例题 的代码实现

```python
import numpy as np

num_hidden_states = 3
num_observations = 2

# 红-->0, 白-->1
observation_sequence = ['红', '白', '红']

obs_map = {'红': 0, '白': 1}

# matrix[t-1, t] ===> t-1 --> t
transition_matrix = np.array([[.5, .2, .3],
                              [.3, .5, .2],
                              [.2, .3, .5]])

# state --> obs, prob
observation_matrix = np.array([[.5, .5],
                               [.4, .6],
                               [.7, .3]])

pi = np.array([.2, .4, .4])

# viterbi
# initialize P(h1, o1) = p(h1) * p(o1|h1)
delta = pi * observation_matrix[:, obs_map[observation_sequence[0]]]
psi = np.zeros(shape=[len(observation_sequence), num_hidden_states],
               dtype=np.uint32)

for i in range(1, 3):
    delta = np.reshape(delta, newshape=[-1, 1])
    P = delta * transition_matrix * np.reshape(
        observation_matrix[:, obs_map[observation_sequence[i]]],
        newshape=[1, -1])
    delta = np.max(P, axis=0)
    psi[i] = np.argmax(P, axis=0)

last_state = np.argmax(delta.reshape(-1))

# state 为 0, 1, 2, 所以打印出来是 2-->2-->2, 书中是 3-->3-->3
for t in reversed(range(3)):
    print(last_state, end='-->')
    last_state = psi[t, last_state]

```

