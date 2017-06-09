# tf.contrib.layers 中的层一览

`tf.contrib.layers`中的接口命名还是挺一致的，搞定一个，其它基本都清楚了。

## conv2d()

```python
layers.conv2d  (inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
```

* `inputs` : 输入 `tensor` 四维

* `num_outputs`: 输出 `channel` 

* `kernel_size`: 可以是 标量 也可以是2个元素的`list` `[kernel_height, kernel_width]` 

* `stride`: 可以是 标量 也可以是2个元素的`list` `[stride_height, stride_width]`

* `padding`: "SAME", "VALID"

*  `data_format`: 指明是 `NHWC` 或者是`HCHW` 默认是`NHWC`

* `rate`: `dilation rate to use  for a'trous convolution`

* `activation_fn`: 卷积操作后跟的激活函数

* `normalizer_fn`:

* `normalizer_params`:

* `weights_initializer`: 权重初始化方法

* `weights_regularizer`: 是否加`weight decay` 默认不加，加的话指定`L1 orL2`

* `biases_initializer`: 偏置初始化方法

* `biases_regularizer`: 是否加`bias decay` 默认不加，加的话指定`L1 orL2`

* `reuse` : 是否 `reuse` 该层

* `variables_collections`: 将`创建的variable`加到哪些`collection`中

* `outputs_collections`: 输出加入到哪些`collections`中

* `trainable`: 如果为`True`， 则将创建的变量加入到`collection` `tf.GraphKeys.TRAINABLE_VARIABLES`中

* `scope`:Optional scope for `variable_scope`.

  * 函数内是用以下方法创建的`Variable scope`意味着，如果不穿`scope` 它会处理命名冲突
    ``` python
    variable_scope.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc
    ```

