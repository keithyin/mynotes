# 保存与恢复变量
在[变量导入与导出](http://blog.csdn.net/u012436149/article/details/52883747)中简单介绍了一下如果保存和恢复变量.最近正好对这方面有需求,所以又查了一些资料,整理一下所了解的东西.

## 如何restore变量的子集,然后使用初始化op初始化其他变量
```python
#想要实现这个功能的话,必须从Saver的构造函数下手
saver=tf.train.Saver([sub_set])
init = tf.initialize_all_variables()
with tf.Session() as sess:
  #这样你就可以使用restore的变量替换掉初始化的变量的值,而其它初始化的值不受影响
  sess.run(init)
  saver.restore(sess,"file")
  # train
  saver.save(sess,"file")
```

**参考资料**
[https://www.tensorflow.org/versions/r0.12/how_tos/variables/index.html#saving-and-restoring](https://www.tensorflow.org/versions/r0.12/how_tos/variables/index.html#saving-and-restoring)
[https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#Saver](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#Saver)
