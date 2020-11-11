# python 时间的两个包

* `import time`
* `import datetime`



# 时间的基本操作 (datetime)

```python
import datetime

# 获取当前时间
now = datetime.datetime.now()

# string 转时间 , "%Y-%m-%d %H:%M:%S"
someDay = datetime.datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')

# 时间转string "%Y-%m-%d %H:%M:%S %a,%b"  a是周几, b是月份
someDay.strftime("%Y-%m-%d %H:%M:%S %a,%b")

# 时间的加减 使用 datetime.timedelta() 来构建 时间delta
someDay-datetime.timedelta(days=2)

# 时间比较
someDay < now

# 两个时间的时间差
someTimeDelta = now-someDay

# 时区转换

# 周几？
d=datetime.datetime(2018,11,11)
t=d.weekday() # 周一是 0
```







# 计时器的基本操作 (time)

