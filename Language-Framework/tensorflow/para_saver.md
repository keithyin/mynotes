###如何使用tensorflow内置的参数导出和导入方法：基础特性
####如果你还在纠结如何保存tensorflow训练好的模型参数，用这个方法就对了
```python
import tensorflow as tf
"""
变量声明，运算声明 例：w = tf.get_variable(name="vari_name", shape=[], dtype=tf.float32)
初始化op声明
"""
#创建saver op
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	#训练模型
    saver.save(sess, "save_path/file_name") #file_name如果不存在的话，会自动创建
```
**现在，训练好的模型参数已经存储好了，我们来看一下怎么调用训练好的参数**
**变量保存的时候，保存的是 变量名：value，键值对。restore的时候，也是根据变量名来进行的**
```python
import tensorflow as tf
"""
变量声明，运算声明
初始化op声明
"""
#创建saver op
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)#在这里，可以执行这个语句，也可以不执行，即使执行了，初始化的值也会被restore的值给override
    saver.restort(sess, "save_path/file_name") #会将已经保存的变量值resotre到 变量中。通过name
```
**更高端的用法，见**[点此跳转](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variables.html)