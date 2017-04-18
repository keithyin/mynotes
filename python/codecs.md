# python codecs
codecs用来读取中文文件
```python
import codecs
file_name = "file_name"
with codecs.open(file_name,mode='rb',coding='gbk') as file:
  for line in file:
    for word in line:
      print word
```
**参考文献**
[https://docs.python.org/3/library/codecs.html](https://docs.python.org/3/library/codecs.html)
