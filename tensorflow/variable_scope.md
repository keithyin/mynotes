# variable scope
**tensorflow 为了更好的管理变量,提供了variable scope机制**
**官方解释:**
Variable scope object to carry defaults to provide to get_variable.

Many of the arguments we need for get_variable in a variable store are most easily handled with a context. This object is used for the defaults.

Attributes:
- name: name of the current scope, used as prefix in get_variable.
- initializer: default initializer passed to get_variable.
- regularizer: default regularizer passed to get_variable.
- reuse: Boolean or None, setting the reuse in get_variable.
- caching_device: string, callable, or None: the caching device passed to get_variable.
- partitioner: callable or None: the partitioner passed to get_variable.
- custom_getter: default custom getter passed to get_variable.
- name_scope: The name passed to tf.name_scope.
- dtype: default type passed to get_variable (defaults to DT_FLOAT).

**可以看出,用variable scope管理get_varibale是很方便的**

## 如何确定 get_variable 的 prefixed name
首先, variable scope是可以嵌套的:
```python
with variable_scope.variable_scope("tet1"):
    var3 = tf.get_variable("var3",shape=[2],dtype=tf.float32)
    print var3.name
    with variable_scope.variable_scope("tet2"):
        var4 = tf.get_variable("var4",shape=[2],dtype=tf.float32)
        print var4.name
#输出为****************
#tet1/var3:0
#tet1/tet2/var4:0
#*********************
```
get_varibale.name 以创建变量的 `scope` 作为名字的prefix
```python
def te2():
    with variable_scope.variable_scope("te2"):
        var2 = tf.get_variable("var2",shape=[2], dtype=tf.float32)
        print var2.name
        def te1():
            with variable_scope.variable_scope("te1"):
                var1 = tf.get_variable("var1", shape=[2], dtype=tf.float32)
            return var1
        return te1() #在scope te2 内调用的
res = te2()
print res.name
#输出*********************
#te2/var2:0
#te2/te1/var1:0
#************************
```
观察和上个程序的不同
```python
def te2():
    with variable_scope.variable_scope("te2"):
        var2 = tf.get_variable("var2",shape=[2], dtype=tf.float32)
        print var2.name
        def te1():
            with variable_scope.variable_scope("te1"):
                var1 = tf.get_variable("var1", shape=[2], dtype=tf.float32)
            return var1
    return te1()  #在scope te2外面调用的
res = te2()
print res.name
#输出*********************
#te2/var2:0
#te1/var1:0
#************************
```

## 其它
`tf.get_variable_scope()` :获取当前scope
`tf.get_variable_scope().reuse_variables()` 共享变量
