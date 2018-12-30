# 令人头疼的编码问题

> * Unicode 是一个符号集，包含世界上所有的 文本符号
>   * 文本符号：书面上可以看到的符号。各种的数学符号，文字都是
> * UTF-8,GB2312： 是个具体的 **编/解 码** 方法，由于计算机中只能存储 `01`，所以编码方法是用来声明，Unicode中的符号在计算机中该怎么存储，即 符号--> 字节流 的映射。解码就是来说明，字节流-->符号的映射。
>   * '你' : 使用 UTF8编码的话，在计算机中存储的就是 `\xe4\xbd\xa0`(16进制)
>   * '你'：使用 gbk 编码的话：在计算机中存储的就是 `\xc4\xe3`(16进制)



## python 中的编码

对 **Unicode 和 UTF8/GBK** 有了了解之后，现在来看 python



**python 中有两种 string（字符串）**

* unicode string:  符号 string，比如 ：`"你好"`
  * **unicode 符号可以用于各个编码之间转化的中介**
  * 如果将 unicode string 写到文件中，**存储形式取决于 `open()` 时候的设置的编码**
  * 如果是 python 脚本文件里面的 unicode string，那就取决于 python脚本 **文件** 的编码
    * `a="你好"` ， 存储之后什么样就取决于 脚本文件的编码了。
* byte string: 字节流，unicode string 编码后的表示。
  *  `u"你好"`用 utf8 编码以后就是`b"\xe4\xbd\xa0\xe5\xa5\xbd"`
* 编码：`encode` ，将 `unicode string` 编码成对应的字节流
* 解码： `decode` ，将字节流解码成 `unicode string`

```python
a = "你好" #这是个 unicode string
a_utf8_byte_str=a.encode("utf8") # b'\xe4\xbd\xa0\xe5\xa5\xbd'
a_gbk_byte_str=a.encode("gbk")   # b'\xc4\xe3\xba\xc3'

a_utf8_byte_str.decode("utf8")   # 你好，将字节流 解码成 unicode string
a_gbk_byte_str.decode("gbk")     # 你好

a_utf8_byte_str.decode("gbk")    # 这就会报错了
```



**当我们在 linux terminal 执行 python 文件时，大体过程如下**

* python 读入python脚本，以 `UTF-8` 的形式解码保存在硬盘上的脚本
  * 脚本在硬盘上二进制的形式保存的
  * 所以如果python脚本文件不是以 `UTF-8` 编码保存的，解析就会报错
* python 解释器开始执行脚本
* 当碰到 `print` 语句的时候，python 同样用 `UTF-8` 编码要输出的 unicode string，然后放到标准输出缓冲区中。
  * 用什么样的编码输出在 python 中是可以设置的。
* linux terminal 按照自己的配置来 decode python输出的字节流`(0001 1101 0011 0001)`
  * 当然如果 linux terminal 的解码方式 和 python 的编码方式不一致的话，那就会报错了