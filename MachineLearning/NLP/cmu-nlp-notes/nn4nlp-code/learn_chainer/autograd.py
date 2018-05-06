import chainer
from chainer import Variable
import numpy as np
a = Variable(np.array([1,2,3], dtype=np.float32))

b = a + 1

b.grad = np.array([1,1,1], dtype=np.float32)

b.backward()
print(a.grad)
b.backward()
print(a.grad)

import torch
from torch.nn import init
