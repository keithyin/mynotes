# 跑 faster-rcnn代码总结，python3



1. basestring 问题，加上下面那句就好了

   ```python
   try:
       unicode = unicode
   except NameError:
       # 'unicode' is undefined, must be Python 3
       str = str
       unicode = str
       bytes = bytes
       basestring = (str,bytes)
   else:
       # 'unicode' exists, must be Python 2
       str = str
       unicode = unicode
       bytes = str
       basestring = basestring
   ```

   ​

2. 各种改 `print`

