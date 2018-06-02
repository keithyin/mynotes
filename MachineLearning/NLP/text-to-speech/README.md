# Text to Speech

* `amp` : 代表能量谱
* `power` : 代表功率谱



## 数据处理细节总结



**log dynamic range compression**

```python
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    控制了 log 之后的取值范围。
    PARAMS
    ------
    x: mel power spectrum, 功率谱 ---> logarithm space
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)
```



**raw-audio ----> melspectogram 过程解释**

```python
# 傅立叶变换, res 得到的是复数
D = stft(data)

# 计算功率谱，或能量谱谱, res 为能量谱，res**2 为功率谱
res = np.abs(D)

# 用 mel 的 bin 来处理傅立叶变换的结果(处理 能量谱， 又称 amp)。
...

# 然后计算分贝， 要指定分贝的最小值，切一下

```

