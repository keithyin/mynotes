# 总览
tf2默认已经切到eager模式了，如果继续使用tf1的思维定势思考tf2有点别扭。如果有pytorch或者mxnet经验的话，那么切换就比较容易了。
tf2 eager模式 和 tf1 的主要不同点有：
* op都是命令式的
* 每次前向都会重新构建计算图

https://www.tensorflow.org/tutorials/quickstart/advanced

* Layer: tf.keras.layers.*
* Loss: tf.keras.losses.*
* Optimizer: tf.keras.optimizers.*
* Metric: tf.keras.metrics.*

tf.Module & tf.keras.layers.Layer & tf.keras.Model
* tf.keras.Model继承自 tf.keras.layers.Layer
* tf.keras.layers.Layer 继承自 tf.Module

tf.Module 与 torch.nn.Module基本完全等价
torch.nn.Module有一点不好用的是，参数的 in_feature 和 out_feature都需要我们自己指定，tf.keras.layers.Layer提供了 in_feature不需要手动指定的功能。



# 输入

# 模型
如果需要自定义Layer，需要继承 `tf.keras.layers.Layer` 这个和 `torch.nn.Module` 基本等价。

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.fc1 = tf.keras.layers.Dense(10)
        self.fc2 = tf.keras.layers.Dense(1)
    def build(self, shape):
        pass # 如果该模型有自己的参数，在这里创建。
    def call(self, x):
        return self.fc2(self.fc1(x))
```
tf.keras.Model用来建模整个模型！tf.keras.layers.Layer用来建模某层。

# 反向传导
反向传导的核心是 tf.GradientTape(), 这个可以简单理解为：在 gradient_tape context 下的计算，会被追踪前向计算图 已为了构建反向传导图。
这里需要注意的一点是，并非 gradient_tape 下的所有计算都记录反向传导图，只有被 watched 的变量参与计算的子图才会构建相应的反向传导图。
什么变量会被watch：在gradient_tape下参与计算的 trainable=True 的Variable & 手动watch的一些tensor
https://www.tensorflow.org/api_docs/python/tf/GradientTape

# 参数的导出与导入
tf.keras.Model, tf.keras.layers.Layer 都继承自 tf.Module.
tf.Module可以按照以下方式保存

### 保存与加载参数
https://www.tensorflow.org/guide/intro_to_modules
```python
chkp_path = 'checkpoint_file_path'
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)
checkpoint.write(chkp_path)
```

```python
new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")
```

### 保存与加载计算图


# 模型持久化

