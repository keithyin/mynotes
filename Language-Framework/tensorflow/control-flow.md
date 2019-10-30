# Control Flow
介绍几个 `tf` 中的 `control flow` `API`，什么是`control flow`呢？就是 `if-else`, `while`, `switchcase`这些。



## tf.cond(pred, fn1, fn2, name=None)

等价于:
```python
res = fn1() if pred else fn2()
```
**注意：pred不能是 python bool， pred是个标量Tensor i.e. `tf.placeholder(dtype=tf.bool, shape=[])`**
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

* exclusive: 如果为True，那么pred至多有一个为True，如果有多余一个，会报错。如果False，则不会检查所有条件。 

```python
import tensorflow as tf

x = tf.constant(0)
y = tf.constant(1)
z = tf.constant(2)

def f1(): return tf.constant(17)
def f2(): return tf.constant(23)
def f3(): return tf.constant(-1)

r = tf.case({tf.less(x, y): f2, tf.less(x, z): f1},
         default=f3, exclusive=False)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(r))
```

## tf.group() 与 tf.tuple()
如果我们有很多 `tensor` 或 `op`想要一起`run`，这时这两个函数就是一个很好的帮手了。
```python
w = tf.Variable(1)
mul = tf.multiply(w, 2)
add = tf.add(w, 2)
group = tf.group(mul, add)
tuple = tf.tuple([mul, add])
# sess.run(group)和sess.run(tuple)都会求Tensor(add)
#Tensor(mul)的值。区别是，tf.group()返回的是`op`
#tf.tuple()返回的是list of tensor。
#这样就会导致，sess.run(tuple)的时候，会返回 Tensor(mul),Tensor(add)的值.
#而 sess.run(group)不会
```

## tf.identity()
[http://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for](http://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for)

## tf.while_loop()
`tf.while_loop(cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)`

`while_loop`可以这么理解
```python
loop_vars = [...]
while cond(*loop_vars):
	loop_vars = body(*loop_vars)	
```

示例：
```python
import tensorflow as tf

a = tf.get_variable("a", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
b = tf.constant(2)

f = tf.constant(6)

# Definition of condition and body
def cond(a, b, f):
    return a < 3
    
def body(a, b, f):
    # do some stuff with a, b
    a = a + 1
    return a, b, f
# Loop, 返回的tensor while 循环后的 a，b，f
a, b, f = tf.while_loop(cond, body, [a, b, f])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run([a, b, f])
    print(res)
```



除了传统的 `tf.Tensor` 和 `IndexedSlices`, `tf.while_loop` 还可以接受 `tf.TensorArray` 作为其输入.

* `Note that while_loop calls cond and body exactly once ` : 所以在里面 创建变量, 其实也就是创建一次. 并不是 循环几次就创建几次.   总之: `while_loop` 只是创建了一个 循环的流.

* `shape_invariants`: 如果不指定的话, 循环的时候需要保证 `loop_vars` 的 `shape` 在 循环的过程中是不会变的.可以通过指定 `shape_invariants` 来放松这种限制

* 

```python
import tensorflow as tf

a = tf.constant(value=0, dtype=tf.float32)
b = tf.constant(value=1, dtype=tf.float32)

def loop_cond(a, b):
    return tf.less(a, 10)

# loop_body 其实只是执行一次(python层面),  其实就是搭了个 op,然后数据循环往里面流 
def loop_body(a, b):
    a = a + 1
    b = b + 1
    # 使用 tf.Variable 会报错
    w = tf.get_variable(name="w", shape=[2, 3], dtype=tf.float32)
    return a, b

a, b = tf.while_loop(loop_cond, loop_body, loop_vars=[a, b])

print(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fetches=[a, b]))

```



## tf.TensorArray

```python
import tensorflow as tf

a = tf.constant(value=0, dtype=tf.int32)
b = tf.constant(value=1, dtype=tf.float32)
c = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

def loop_cond(a, b, c):
    return tf.less(a, 10)


def loop_body(a, b, c):
  # 一定得是 c = c.write(a, b), 不要傻傻的写成 c.write(a, b) 就以为完事了
  # 如果只是写 c.write(a, b), 会报Object was never used 错误
  # 虽然 c.write(a, b).mark_used() 会防止报错, 但是输出行为却是错的
    c = c.write(a, b)
    a = a + 1
    b = b + 1
    return a, b, c


a, b, c = tf.while_loop(loop_cond, loop_body, loop_vars=[a, b, c])

c = c.stack()
print(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(fetches=[a, b, c]))
```





## 参考资料

[https://www.tensorflow.org/versions/r0.12/api_docs/python/control_flow_ops/control_flow_operations#cond](https://www.tensorflow.org/versions/r0.12/api_docs/python/control_flow_ops/control_flow_operations#cond)
