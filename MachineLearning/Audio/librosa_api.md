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
# 返回值 shape：[1 + n_fft/2, t]
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
# power = 1.0 能量谱 power = 2.0 功率谱
# y ：audio time-series shape=[n, ]
# S : spectrum [d, t], stft 计算得到的 功率谱， 
librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, power=2.0, **kwargs)

```

