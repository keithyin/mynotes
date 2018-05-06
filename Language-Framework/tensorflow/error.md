# tensorflow 错误总结

## TypeError: Expected binary or unicode string
**原因：**
函数参数问题
```python
for grads_per_var in zip(*(self.grads_)):
    grads_per_var = tf.convert_to_tensor(grads_per_var)
    averaged_grads.append(tf.reduce_mean(grads_per_var))
return averaged_grads
```
