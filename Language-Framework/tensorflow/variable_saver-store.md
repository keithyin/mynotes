# 变量保存与恢复
在[变量导入与导出](http://blog.csdn.net/u012436149/article/details/52883747)中，介绍了`saver.saver()`与`saver.restore()`的基本用法，但是那个远远达不到日常使用的需求。下面根据日常使用场景给出一些框架。

## 几个op
先介绍几个`op`函数
```python
tf.variables_initializer(var_list, name='init')
# 这个op用来初始化var_list中的变量（对象），如果为空，啥都不干

tf.report_uninitialized_variables(var_list=None,name='report_uninitialized_variables')
# var_list : 需要检查的参数列表，默认为所有变量对象，记住（是对象而不是名字）
# 返回未被初始化变量的名字 1-D tensor
```
## 如何恢复模型变量的一个子集，然后对剩下的变量进行初始化op
