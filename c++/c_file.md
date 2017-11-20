# C 文件操作 API 总结

## 什么是文件

一个文件（file）通常是磁盘上的一段命名存储区。`C` 将文件看成是连续的字节序列，其中每一个字节都可以单独的读取。

ANSI C 提供了文件的两种视图：

* 文本视图：**程序看到内容与文件的内容有可能不同**
* 二进制视图：**文件中的每个字节都可以为程序所访问**



## 文件操作

### 一、 读文件

[http://www.cplusplus.com/reference/cstdio/fopen/](http://www.cplusplus.com/reference/cstdio/fopen/)

```c++
FILE * fopen ( const char * filename, const char * mode ); // <stdio.h>, <cstdio> 
```

* 打开文件， 名字为 `filename`, 模式为 `mode`
* `mode` : `r, w, a`， `r+, w+, a+`, `rb, wb, ab`, `rb+, wb+, ab+` 对于 `unix` 来说，加不加 `b` 是一样的 

### 二、 关闭文件

[http://www.cplusplus.com/reference/cstdio/fclose/](http://www.cplusplus.com/reference/cstdio/fclose/)

```c
int fclose ( FILE * stream );
```

* 关闭文件， stream 是 open 是返回的指针， 如果关闭成功返回0, 否则，返回 EOF。可以通过是否返回 0 来判断是否关闭成功。

### 三、读写文件

* `getc()` --- `putc()`
* `fgets()` --- `fputs()`

[http://www.cplusplus.com/reference/cstdio/getc/?kw=getc](http://www.cplusplus.com/reference/cstdio/getc/?kw=getc)

[http://www.cplusplus.com/reference/cstdio/putc/?kw=putc](http://www.cplusplus.com/reference/cstdio/putc/?kw=putc)

```c
int getc ( FILE * stream ); // 从 stream 中 取出一个 char， 如果到达文件结尾， 会返回 EOF
int putc ( int character, FILE * stream ); // 像 stream 中写入一个 char， 位置由 stream 的内部指针决定。
```



[http://www.cplusplus.com/reference/cstdio/fgets/?kw=fgets](http://www.cplusplus.com/reference/cstdio/fgets/?kw=fgets)

[http://www.cplusplus.com/reference/cstdio/fputs/](http://www.cplusplus.com/reference/cstdio/fputs/)

```c
char * fgets ( char * str, int num, FILE * stream );// 不会丢掉换行符，会向末尾加一个 空字符构成字符串
int fputs ( const char * str, FILE * stream ); // 不会添加换行符，
```



### 四、任意位置操作文件

* fseek
* ftell

[http://www.cplusplus.com/reference/cstdio/fseek/?kw=fseek](http://www.cplusplus.com/reference/cstdio/fseek/?kw=fseek)

`FILE` 对象中 有个属性(position indicator)是 记录着当前要访问的 位置。`fseek` 就是用来操作这个属性的。

`getc` 时， 会返回当前位置的 char， 然后 position indicator 加一。

```c
int fseek ( FILE * stream, long int offset, int origin );
```

* 设置 position indicator
* offset : 可正可负
* origin ： `SEEK_SET `(文件的开头) , `SEEK_CUR`( 当前位置), `SEEK_END`（文件结尾）

[http://www.cplusplus.com/reference/cstdio/ftell/](http://www.cplusplus.com/reference/cstdio/ftell/)

```c
long int ftell ( FILE * stream ); // 获取当前位置
```



## 参考资料

C Primer Plus

