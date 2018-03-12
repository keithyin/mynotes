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







## 参考资料

[https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a](https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a)

