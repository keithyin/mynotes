# control flow

## tf.cond(pred, fn1, fn2, name=None)
等价于:
```python
res = fn1() if pred else fn2()
```
**注意：pred不能使 python bool， pred是个标量Tensor**
官网例子
```python
z = tf.mul(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

## tf.case(pred_fn_pairs, default, exclusive=False, name='case')
`pred_fn_pairs`:以下两种形式都是正确的
1. [(pred_1, fn_1), (pred_2, fn_2)]
2. {pred_1:fn_1, pred_2:fn_2}

`tf.case()`等价于:
```python
if pred_1:
  return fn_1()
elif pred_2:
  return fn_2()
else:
  return default()
```

## tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)
