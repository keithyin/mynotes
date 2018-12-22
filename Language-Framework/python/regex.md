# python 正则表达式

* 检查是字符串是否有 **子串** 满足正则表达式的 `pattern`



```python
import re

# 以下两种写法是等价的
reg = re.compile(r"pattern")
reg.match("hello world")

re.match(r"pattern", "hello world") 
```



```python
res = re.match(pattern, string, flags=0)
#字符串以开头为始的子串 是否能匹配正则表达式。
#返回_sre.SRE_Match对象，如果没有能匹配上的字串，则返回None。
# 如果匹配的话，res.string可以获得原始的字符串，并不是匹配的字符串 
```



```python
re.sub(pattern, repl, string, count=0, flags=0)

# 找到string中 能和 pattern 匹配的所有子串，并将其用repl替换。
# 可选参数：count 是模式匹配後替换的最大次数；count 必须是非负整数。
# 缺省值 是 0 表示替换所有的匹配。如果无匹配，字符串将会无改变地返回。
# 如果有匹配,则返回替换后的字符串
# pattern='he$' 尾部匹配
# pattern='^he' 头部匹配，等价于match
```



```python
re.findall(pattern,string)
# 从 string中找到所有 匹配 pattern的子串，作为列表返回
# 如果没有匹配的话，返回空数组，可用来当做if的判断条件,空数组为False
# pattern='he$' 尾部匹配
# pattern='^he' 头部匹配，等价于match
```



```python
re.search(pattern, string, pos=0, endpos=-1)
#顾名思义，查找，如果找到返回一个match对象，找不到，返回None。(找到第一个就返回)
# pattern='he$' 尾部匹配
# pattern='^he' 头部匹配，等价于match
```



## Grouping

* 将几个字符放在一组当作一个单元就叫做grouping，语法 `(c1c2)`
* `(abc)` :  现在 abc 就是一个单元了，`(abc)*` `(abc)+` 就是以 `abc` 为单位的了
* python 也支持查看 string 的字串匹配中，匹配pattern组的结果

```python
reg = re.compile(r"(hel*).*(is)")
res = reg.match("hello he is good")
res.group(0) # 返回 满足整个pattern 的 string字串
res.group(1) #返回满足的字串中，哪些 substr 满足第一个 group
res.group(2) # 返回满足的字串中，哪些 substr 满足第二个 group
res.groups() # 不包含的 group() 的结果。
print(res)

"""
hello he is
hell
is
('hell', 'is')
"""
```





## 参考资料

[https://docs.python.org/2.7/library/re.html](https://docs.python.org/2.7/library/re.html)

[https://docs.python.org/2.7/howto/regex.html#regex-howto](https://docs.python.org/2.7/howto/regex.html#regex-howto)