# 音频数据处理



## bit-depth

在使用 pulse-code modulation (PCM) 的数字音频中, **bit depth** is the number of [bits](https://en.wikipedia.org/wiki/Bit) of information in each [sample](https://en.wikipedia.org/wiki/Sampling_(signal_processing)), and it directly corresponds to the **resolution** of each sample. Examples of bit depth include [Compact Disc Digital Audio](https://en.wikipedia.org/wiki/Compact_Disc_Digital_Audio), which uses 16 bits per sample, and [DVD-Audio](https://en.wikipedia.org/wiki/DVD-Audio) and [Blu-ray Disc](https://en.wikipedia.org/wiki/Blu-ray_Disc) which can support up to 24 bits per sample.



## mono

双声道的音频合成单声道的, 就是双声道的第一维求个平均.



## .wav 文件

* `.wav` 文件中保存的是 采样后的数据. 


* 采样频率代表着, 多少个数据表示 **1s** 的时长.



**python 读取 .wav 文件**

```python
import scipy.io.wavfile as wav

# freq 表示 wav 文件的采样频率
# audio_data , 如果是单声道 shape 为 (num_point,) 如果是双声道 为 (num_point, 2)
freq, audio_data = wav.read(first_audio)

"""
=====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
=====================  ===========  ===========  =============
32-bit floating-point  -1.0         +1.0         float32
32-bit PCM             -2147483648  +2147483647  int32
16-bit PCM             -32768       +32767       int16
8-bit PCM              0            255          uint8
=====================  ===========  ===========  =============
"""
```



**librosa读取音频文件**

```python
def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best')
# sr:读取出来的采样频率
# mono: 是否转成单声道
# offset:开始时间, seconds
# duration: 持续时间, seconds
# 返回值: 归一化到 [-1, 1] 之间. int32类型就除 2^16, int16类型就除 2^8
```





**torchaudio 读取音频文件**

```python
torchaudio.load(filepath, out=None, normalization=None)
# normalization : bool or number, 如果为 true, 除2^31(假设是16-bit depth, int32类型), 
# 如果是数字的话,除以指定的数字
# 如果为 True, 和 librosa 效果一样.
```



## 信号处理

**傅立叶频谱**

```python
scipy.signal.spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
# 使用傅立叶变换计算频谱
# fs: x数据的采样频率
# nperseg: n-per-segment
# noverlap: 两个 segment 之间的重叠
# nfft: 如果为None, FFT的长度=nperseg, 如果有值, 则表示FFT的长度
```



```python
sample_rate, samples = wav.read(first_audio)
freqs, times, spec = signal.spectrogram(samples, sample_rate, window="hann", nperseg=int(20 / 1e3 * sample_rate),noverlap=int(10 / 1e3 * sample_rate), detrend=False)
# freqs : 表示每个值所代表的频率, shape (n,)
# times: 表示每个 timestep 的时间, shape (m,)
# spec: 频率的幅值, shape (n,m)
# 可视化的时候,一般都是求 log(spec)
```



**Mel power spectrogram**

> 使用 librosa 库

```python
# 这里读取 .wav 文件就不能再用 scipy.io.wavfile, 得用 librosa.load, 它读出来的是归一化后的.
# S: [num_mels, time], 
S = feature.melspectrogram(samples, sr=sample_rate, n_mels=228)

# log_S: shape [num_mels, time], (-80db, 0db)
log_S = librosa.power_to_db(S, ref=np.max)
```



**计算 MFCC**

```python
librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs)
# 计算MFCC
# y: ndarray (n,) 或者 None, 原始音频信号
# sr: 采样频率, None的话使用文本的采样频率
# S: log-power Mel spectrogram (d,t) 或者 None
# n_mfcc: mfcc 数量
```

```python
# 使用原始音频数据计算 MFCC
y, sr = librosa.load(librosa.util.example_audio_file())
librosa.feature.mfcc(y=y, sr=sr)
"""
array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
       [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
       ...,
       [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
       [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])
"""
```

```python
# 使用计算好的 Mel spectrogram 计算 MFCC
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)
librosa.feature.mfcc(S=librosa.power_to_db(S))
"""
array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
       [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
       ...,
       [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
       [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])
"""
```



## 关于声音

[https://zhuanlan.zhihu.com/p/22821588, 什么是分贝](https://zhuanlan.zhihu.com/p/22821588)

**分贝**

> 定义为两个数值的对数比率, 这两个数分别是测量值和参考值(也称为基准值). 存在两种定义情况.

* 功率之比

$$
1db = 10\log_{10}\Bigr(\frac{W}{W_o}\Bigr)
$$

**waveform**









## 参考资料

[https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a)

[https://www.kaggle.com/davids1992/speech-representation-and-data-exploration](https://www.kaggle.com/davids1992/speech-representation-and-data-exploration)

[https://librosa.github.io/librosa/index.html](https://librosa.github.io/librosa/index.html)