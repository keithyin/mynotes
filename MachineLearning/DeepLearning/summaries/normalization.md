# Normalization 总结

* batch-norm
* layer-norm
* instance-norm



**BatchNorm (用在CNN和FC上)** 

* 训练的时候: 计算每个 `特征的特征值` 在整个 batch 上的均值和方差. 同时会更新 `moving_mean, moving_variance`
* 推断的时候: 用 `moving-mean` 和` moving-variance`，用来代替整个数据集的均值与方差
  * 因为推断的情况可能是 一个个 样本进行推断

假设输入数据 `shape` 为 `[B, C, H, W]`,  

```python
# 训练时候的计算方法是：
mean = np.mean(data, axis=[0,2,3])
std = np.std(data, axis=[0,2,3])
out = (data - mean) / std * gamma + beta # gamma: [C], beta: [C]

# 推断时
out = (data - moving_mean) / moving_std * gamma + beta
```



**InstanceNorm(用在CNN上,用来获得feature map 的统计特性)**

* 不在 batch 维度上求均值, 计算每个样本的每个 channel  上的均值(feature map 的统计特性)。
* 也不需要 `moving_mean, moving_variance`

```python
mean = np.mean(data, axis=[2, 3])
std = np.std(data, axis=[2, 3])
out = (data - mean) / std * gamma + beta # gamma: [C], beta: [C]
```



**LayerNorm** (用在 RNN 上)

* 在RNN的每个时间步上使用 LayerNorm
* 确认一下在 self-attention中如何使用LayerNorm
* **计算每个样本所有特征的均值** , BN是计算,各个特征在mini-batch上的均值
* 不会保存 moving-mean 和 moving-variance

```python
mean = np.mean(data, axis=[1, 2, 3])
std = np.std(data, axis=[1, 2, 3])
out = (data-mean)/std * gamma + beta # gamma: [C, H, W] 
# 每个点都会学习一个 gamma 和 beta
```





## 参考资料

[https://stackoverflow.com/questions/45463778/instance-normalisation-vs-batch-normalisation](https://stackoverflow.com/questions/45463778/instance-normalisation-vs-batch-normalisation)