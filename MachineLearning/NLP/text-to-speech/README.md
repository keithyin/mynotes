# Text to Speech



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

