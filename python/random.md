# python numpy random

```python
np.random.rand(d0,d1,...,dn)
#Create an array of the given shape and populate it
#with random samples from a uniform distribution over
# [0, 1).

np.random.rand(3,2)
#输出
array([[ 0.14022471,  0.96360618],  #random
       [ 0.37601032,  0.25528411],  #random
       [ 0.49313049,  0.94909878]]) #random
```
```python
np.random.randn(d0,d1,...,dn)
#Create an array of the given shape and populate it
#with random samples from a gaussian(norm,(0,1))
#distribution.
```

```python
np.random.randint(low, high, size, dtype)
#上下范围内随机选取整数，包括low，不包括high
np.random.randint(0,3,10)
# array([0, 2, 2, 1, 1, 2, 0, 0, 1, 0])
```
