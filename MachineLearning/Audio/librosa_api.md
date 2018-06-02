# librosa api 总结

`mono`: 是不是将数据转换成单声道





**读取和保存**

```python
# 一般音频文件waveform中每个采样点都是用 16-bit 保存的。所以此函数返回的结果是 是除2^15进行归一化的
# 如果原始音频使用 float 保存的，结果会截断 到 [-1., 1.]
librosa.core.load(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=<class 'numpy.float32'>, res_type='kaiser_best')

# 如果y必须是浮点数， 当norm=False，仅仅是将 y中的值写到文件中去，
# 如果 norm=True，则会多一步转换（归一化操作），再写 y = y / max([max(y), abs(min(y)))。
# 如果 y 本身是 [-1., 1.] 内，norm 设置为 False，如果不是 [-1., 1.] norm 还是设置成 True 比较 
# 好。
librosa.output.write_wav(path, y, sr, norm=False)
```



**傅立叶变换与逆傅立叶变换**

```python
# short time fourier transform, 返回 复数！！！
# 滑动窗口撸，win_length: 滑动窗口大小，hop_length: 一次平移多少。
# 返回值 shape：[1 + n_fft/2, t] t = data_length // hop_length
# t = len(data) // hop_length
librosa.core.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')

# stft 的逆变换，给定频谱（不是功率谱哦），逆变换到 原始波形。
librosa.core.istft(stft_matrix, hop_length=None, win_length=None, window='hann', center=True, dtype=<class 'numpy.float32'>, length=None)


def stft_istft():
    window = 1000
    hop_length = 500
    data, sr = librosa.load(filename, sr=None)
    print(data)
    
    spec = librosa.stft(data, n_fft=window, hop_length=hop_length)
    i_data = librosa.istft(spec, hop_length=hop_length, win_length=window)
    print(i_data)
    # i_data 与 data 一样。
```



**mel 谱**

```python
# power = 1.0 能量谱,  power = 2.0 功率谱
# y ：audio time-series shape=[n, ]
# S : spectrum [d, t], stft 计算得到的 功率谱， 
librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, power=2.0, **kwargs)
```



**分贝**

```python
# 功率谱 转 db
librosa.core.power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0)
"""
这个式子用来计算： 10*log10(S/ref)
ref : 因为 分贝是个相对单位， ref 就是对应相对的那个值。
amin : 对 功率进行 clip。
top_db: 决定了会对 分贝的最小值 进行 clip，保证 max_db - min_db <= top_db
"""
```

* 关于 `amin` : 一般设置为 `10^-5` 即可，因为人类听力阈值就是 `10^-5 Pa`(amplitude), 如果是功率的话，就平方一下即可。`power_to_db` 的就是功率的，求个平方，就是 `10^-10` 了。**这个是用来对 power/amplitude 进行 clip 的。**
* 关于 `ref` ：关于听力的分贝计算， ref 是 `10^-5` （amplitude），`1e-10`（power），得到的是正值。其实当 `amplitude=0.2` 的时候，按照 `ref=1e-5` 来计算的话，就已经是 `80db`了，非常吵了。当我们将 `ref=0.2` 的话，那么得到的大部分分贝就是 `0` 以下了。
* 关于 `top_db` ： 这个是限制感兴趣 `db` 区间范围的。

> 在进行深度学习模型训练的时候，通常会将得到的 db 进行归一化一下。到 [0, 1] 的区间内。

