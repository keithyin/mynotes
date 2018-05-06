# nce_loss

```python
tf.nn.nce_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=False,
    partition_strategy='mod',
    name='nce_loss'
)
# noise-contrastive estimation training loss
```

* **weights: ** tensor of shape [num_classes, dim], 是 word-embedding
* **bias: ** tensor of shape [num_classes], 类别的 偏置
* **labels**: tensor of shape [batch_size, num_true], 数据真实的 标签, num_true 意味这个 loss 可以用在多标签任务上.
* **inputs: ** tensor of shape [batch_size, dim] , 神经网络的输出, nce_loss 的输入
* **num_sampled: ** 每个 batch 要采多少个样本
* **num_classes**: 类别的数量
* **num_true: ** 每个样本的标签数量
* ​