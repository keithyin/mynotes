# Normalization 总结

* batch-norm
* layer-norm
* instance-norm



**BatchNorm**

* 有 moving-mean 和 moving-variance，用来估计整个数据集的均值与方差

假设输入数据 `shape` 为 `[B, C, H, W]`,  

```python
# 训练时候的计算方法是：
mean = np.mean(data, axis=[0,2,3])
std = np.std(data, axis=[0,2,3])
out = (data - mean) / std * gamma + beta # gamma: [C], beta: [C]

# 推断时
out = (data - moving_mean) / moving_std * gamma + beta
```



**InstanceNorm**

* 不在 batch 维度上求均值

```python
mean = np.mean(data, axis=[2, 3])
std = np.std(data, axis=[2, 3])
out = (data - mean) / std * gamma + beta # gamma: [C], beta: [C]
```





**LayerNorm**

* **样本** 的每一个点
* 不会保存 moving-mean 和 moving-variance

```python
mean = np.mean(data, axis=[1, 2, 3])
std = np.std(data, axis=[1, 2, 3])
out = (data-mean)/std * gamma + beta # gamma: [C, H, W] 
# 每个点都会学习一个 gamma 和 beta
```





