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



## 模型数据预处理过程

```python
# 1. 对 raw-audio 进行 rescale , 使得所有的音频都在同一个 volume 下。
raw_audio = raw_audio / np.max(np.abs(raw_audio)) * 0.999

# 2. 傅立叶变换，或者 mel-spectrogram
# 对于 傅立叶变换：然后求 amplitude：得到的值的范围是 [2e-5, 20] 附近，clip 到这个区间。
# 对于 mel-amplitude：得到的值取值范围是 [5e-6, 0.5] 附近， clip 到这个区间内。
spec = np.abs(librosa.stft(raw_audio, n_fft=window, hop_length=hop_length))
mel_bank = librosa.filters.mel(sr=sr, n_fft=window, n_mels=128)
mel_spec = np.dot(mel_bank, spec) # 可以 clip 一下。

# 3. 这时候就要求 db（log space） 了
# 对于 spec : ref 就要设置成 20, 因为希望得到的 db 最大值 为0
# 对于 mel_spec : ref 设置成 .5 即可，得到同样的效果
# 无论 spec 还是 mel_spec 得到最小值都是 -120
db = librosa.amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0) 
# ref 对应设置一下， amin就是切 amplitude 最小值的，可以对应设置一下，top_db，这个设置成 140 就不会影响结果了。

# 4. 对 db 进行归一化 [0, 1]
# 归一化 [0,1] 区间内
db = (db + 120) / 120
```

