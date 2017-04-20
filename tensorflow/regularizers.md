# tensorflow Regularizers
在损失函数上加上正则项是防止过拟合的一个重要方法,下面介绍如何在`TensorFlow`中使用正则项.

`tensorflow`中对参数使用正则项分为两步:
1. 创建一个正则方法(函数/对象)
2. 将这个正则方法(函数/对象),应用到参数上

## 如何创建一个正则方法函数
### tf.contrib.layers.l1_regularizer(scale, scope=None)
返回一个用来执行`L1`正则化的函数,函数的签名是`func(weights)`.
参数:
- scale: 正则项的系数.
- scope: 可选的`scope name`

### tf.contrib.layers.l2_regularizer(scale, scope=None)
返回一个执行`L2`正则化的函数.
### tf.contrib.layers.sum_regularizer(regularizer_list, scope=None)
返回一个可以执行多种(个)正则化的函数.意思是,创建一个正则化方法,这个方法是多个正则化方法的混合体.

参数:
regularizer_list: regulizer的列表

**已经知道如何创建正则化方法了,下面要说明的就是如何将正则化方法应用到参数上**
## 应用正则化方法到参数上
### tf.contrib.layers.apply_regularization(regularizer, weights_list=None)
先看参数
- regularizer:就是我们上一步创建的正则化方法
- weights_list: 想要执行正则化方法的参数列表,如果为`None`的话,就取`GraphKeys.WEIGHTS`中的`weights`.

函数返回一个标量`Tensor`,同时,这个标量`Tensor`也会保存到`GraphKeys.REGULARIZATION_LOSSES`中.这个`Tensor`保存了计算正则项损失的方法.

> `tensorflow`中的`Tensor`是保存了计算这个值的路径(方法),当我们run的时候,`tensorflow`后端就通过路径计算出`Tensor`对应的值

现在,我们只需将这个正则项损失加到我们的损失函数上就可以了.

## 其它
在使用`tf.get_variable()`和`tf.variable_scope()`的时候,你会发现,它们俩中有`regularizer`形参.如果传入这个参数的话,那么`variable_scope`内的`weghts`的正则化损失,或者`weight`的正则化损失就会被添加到`GraphKeys.REGULARIZATION_LOSSES`中.

## 参考资料
[https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/regularizers](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/regularizers)
