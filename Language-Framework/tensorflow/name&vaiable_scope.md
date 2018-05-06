# name&variable scope
水平有限,如有错误,请指正!

**在tensorflow中,有两个scope, 一个是name_scope一个是variable_scope,这两个scope到底有什么区别呢?**
先看第一个程序:
```python
with tf.name_scope("hello") as name_scope:
    arr1 = tf.get_variable("arr1", shape=[2,10],dtype=tf.float32)

    print name_scope
    print arr1.name
    print "scope_name:%s " % tf.get_variable_scope().original_name_scope
```
输出为:
hello/
arr1:0
scope_name:

可以看出:
- tf.name_scope() 返回的是 一个string,"hello/"
- 在name_scope中定义的variable的name并没有 "hello/"前缀
- tf.get_variable_scope()的original_name_scope 是空

第二个程序:
```python
with tf.variable_scope("hello") as variable_scope:
    arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)

    print variable_scope
    print arr1.name
    print tf.get_variable_scope().original_name_scope
    #tf.get_variable_scope() 获取的就是variable_scope

    with tf.variable_scope("xixi") as v_scope2:
        print tf.get_variable_scope().original_name_scope
        #tf.get_variable_scope() 获取的就是v _scope2
```
输出为:
<tensorflow.python.ops.variable_scope.VariableScope object at 0x7fbc09959210>
hello/arr1:0
hello/
hello/xixi/

可以看出:
- tf.variable_scope() 返回的是一个 op对象
- variable_scope中定义的variable 的name加上了"hello/"前缀
- tf.get_variable_scope()的original_name_scope 是 嵌套后的scope name

**对比两个程序可以看出:**
- name_scope对 get_variable()创建的变量 的名字不会有任何影响,而,在vairiable下创建的会被加上前缀.
- tf.get_variable_scope() 返回的只是 variable_scope,不管 name_scope.所以以后我们在使用tf.get_variable_scope().reuse_variables() 时可以无视name_scope

## name_scope可以用来干什么
典型的 TensorFlow 可以有数以千计的节点，如此多而难以一下全部看到，甚至无法使用标准图表工具来展示。为简单起见，我们为变量名划定范围，并且可视化把该信息用于在图表中的节点上定义一个层级。默认情况下， 只有顶层节点会显示。下面这个例子使用tf.name_scope在hidden命名域下定义了三个操作:
```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
  print a.name
  print W.name
  print b.name
```
结果是得到了下面三个操作名:

hidden/alpha
hidden/weights
hidden/biases
**name_scope 是给op_name加前缀, variable_scope是给variable_name加前缀.也许有人会问,get_variable()返回的不也是op吗?我们可以这么理解,只有get_variable().name打印出来的是variable_name**

## variable_scope可以用来干什么
variable_scope 用来管理 variable 详见[variable_scope](http://blog.csdn.net/u012436149/article/details/53018924)
