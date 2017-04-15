# merge_all引发的血案

1. 在训练深度神经网络的时候，我们经常会使用Dropout，然而在`test`的时候，需要把`dropout`撤掉.为了应对这种问题，我们通常要建立两个模型，让他们共享变量。[详情](http://blog.csdn.net/u012436149/article/details/53843158).
2. 为了使用`Tensorboard`来可视化我们的数据，我们会经常使用`Summary`，最终都会用一个简单的`merge_all`函数来管理我们的`Summary`

## 错误示例
当这两种情况相遇时，`bug`就产生了，看代码：
```python
import tensorflow as tf
import numpy as np
class Model(object):
    def __init__(self):
        self.graph()
        self.merged_summary = tf.summary.merge_all()# 引起血案的地方
    def graph(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None,1])
        w = tf.get_variable("w",shape=[1,1])
        self.predict = tf.matmul(self.x,w)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.label-self.predict),axis=1))
        self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        tf.summary.scalar("loss",self.loss)
def run_epoch(session, model):
    x = np.random.rand(1000).reshape(-1,1)
    label = x*3
    feed_dic = {model.x.name:x, model.label:label}
    su = session.run([model.merged_summary], feed_dic)
def main():
    with tf.Graph().as_default():
        with tf.name_scope("train"):
            with tf.variable_scope("var1",dtype=tf.float32):
                model1 = Model()
        with tf.name_scope("test"):
            with tf.variable_scope("var1",reuse=True,dtype=tf.float32):
                model2 = Model()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            run_epoch(sess,model1)
            run_epoch(sess,model2)
if __name__ == "__main__":
    main()
```
运行情况是这样的： 执行`run_epoch(sess,model1)`时候，程序并不会报错，一旦执行到`run_epoch(sess,model1)`，就会报错（错误信息见文章最后）。
## 错误原因
看代码片段：
```python
class Model(object):
    def __init__(self):
        self.graph()
        self.merged_summary = tf.summary.merge_all()# 引起血案的地方
...
with tf.name_scope("train"):
    with tf.variable_scope("var1",dtype=tf.float32):
        model1 = Model() # 这里的merge_all只是管理了自己的summary
with tf.name_scope("test"):
    with tf.variable_scope("var1",reuse=True,dtype=tf.float32):
        model2 = Model()# 这里的merge_all管理了自己的summary和上边模型的Summary
```
由于`Summary`的计算是需要`feed`数据的，所以会报错。

## 解决方法
我们只需要替换掉`merge_all`就可以解决这个问题。看代码
```python
class Model(object):
    def __init__(self，scope):
        self.graph()
        self.merged_summary = tf.summary.merge(
        tf.get_collection(tf.GraphKeys.SUMMARIES,scope)
        )
...
with tf.Graph().as_default():
    with tf.name_scope("train") as train_scope:
        with tf.variable_scope("var1",dtype=tf.float32):
            model1 = Model(train_scope)
    with tf.name_scope("test") as test_scope:
        with tf.variable_scope("var1",reuse=True,dtype=tf.float32):
            model2 = Model(test_scope)
```
关于`tf.get_collection`[地址](http://blog.csdn.net/u012436149/article/details/53894354)

## 当有多个模型时，出现类似错误，应该考虑使用的方法是不是涉及到了其他的模型


## error
`tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor 'train/var1/Placeholder' with dtype float
	 [[Node: train/var1/Placeholder = Placeholder[dtype=DT_FLOAT, shape=[], _device="/job:localhost/replica:0/task:0/gpu:0"]()]]`
