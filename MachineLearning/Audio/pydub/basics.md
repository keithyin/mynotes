# pydub 基本用法



## 文件读取与保存

**读出来的是 bytestring**

```python
from pydub import AudioSegment

song = AudioSegment.from_wav("never_gonna_give_you_up.wav")
song = AudioSegment.from_mp3("never_gonna_give_you_up.mp3")
ogg_version = AudioSegment.from_ogg("never_gonna_give_you_up.ogg")
flv_version = AudioSegment.from_flv("never_gonna_give_you_up.flv")

mp4_version = AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
wma_version = AudioSegment.from_file("never_gonna_give_you_up.wma", "wma")
aac_version = AudioSegment.from_file("never_gonna_give_you_up.aiff", "aac")
```

**转换到 numpy**

```python
import numpy as np

data = np.array(song.get_array_of_samples()) 
```

**保存**

```python
awesome.export("mashup.mp3", format="mp3")
awesome.export("mashup.mp3", format="mp3", tags={'artist': 'Various artists', 'album': 'Best of 2011', 'comments': 'This album is awesome!'})
awesome.export("mashup.mp3", format="mp3", bitrate="192k")

# Use preset mp3 quality 0 (equivalent to lame V0)
awesome.export("mashup.mp3", format="mp3", parameters=["-q:a", "0"])

# Mix down to two channels and set hard output volume
awesome.export("mashup.mp3", format="mp3", parameters=["-ac", "2", "-vol", "150"])
```





## 音频切片

**音频切片是工作在 milliseconds ** 数量级的

```python
ten_seconds = 10 * 1000

first_10_seconds = song[:ten_seconds]

last_5_seconds = song[-5000:]
```



## 音量调节

```python
# boost volume by 6dB
beginning = first_10_seconds + 6

# reduce volume by 3dB
end = last_5_seconds - 3
```



## 音频连接

```python
without_the_middle = beginning + end
```





## AudioSegment

```python
self.channels = raw.getnchannels() # 单声道/双声道

# 样本宽度, 1: 8-bit audio, 2: 16-bit audio 4:32-bit audio 
self.sample_width = raw.getsampwidth() 
self.frame_rate = raw.getframerate() # 采样频率
self.frame_width = self.channels * self.sample_width
```



