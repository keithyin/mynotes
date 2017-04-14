# 如何使用Supervisor

## 使用方法
在不使用`Supervisor`的时候，我们的代码经常是这么组织的
```python
variables
...
ops
...
summary_op
...
merge_all_summarie
saver
init_op

with tf.Session() as sess:
  writer = tf.tf.train.SummaryWriter()
  sess.run(init)
  saver.restore()
  for ...:
    train
    merged_summary = sess.run(merge_all_summarie)
    writer.add_summary(merged_summary,i)
  saver.save
```
下面介绍如何用`Supervisor`来改写上面程序
```python
import tensorflow as tf
a = tf.Variable(1)
b = tf.Variable(2)
c = tf.add(a,b)
update = tf.assign(a,c)
tf.scalar_summary("a",a)
init_op = tf.initialize_all_variables()
merged_summary_op = tf.merge_all_summaries()
sv = tf.train.Supervisor(logdir="/home/keith/tmp/",init_op=init_op) #logdir用来保存checkpoint和summary
saver=sv.saver #创建saver
with sv.managed_session() as sess: #会自动去logdir中去找checkpoint，如果没有的话，自动执行初始化
    for i in xrange(1000):
        update_ = sess.run(update)
        print update_
        if i % 10 == 0:
            merged_summary = sess.run(merged_summary_op)
            sv.summary_computed(sess, merged_summary,global_step=i)#就像add_summary
        if i%100 == 0:
            saver.save(sess,logdir="/home/keith/tmp/",global_step=i)
```

## 总结
从上面代码可以看出，`Supervisor`帮助我们处理一些事情
（1）自动去checkpoint加载数据或初始化数据
（2）自身有一个`Saver`，可以用来保存checkpoint
（3）有一个`summary_computed`用来保存`Summary`
所以，我们就不需要：
（1）手动初始化或从`checkpoint`中加载数据
（2）不需要创建`Saver`，使用`sv`内部的就可以
（3）不需要创建`summary writer`
