import chainer
from chainer import Variable
import numpy as np
a = Variable(np.array([1,2,3], dtype=np.float32))
print(a)