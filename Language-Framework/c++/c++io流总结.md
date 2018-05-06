# C++ I/O 流总结

c++语言不直接处理输入输出，而是通过一族定义在标准库中的类型来处理I/O。这些类型支持从设备读取数据、向设备写入数据的IO操作。设备可以是文件、控制台窗口等。

IO库定义了读写内置类型的操作。



| 头文件      | 类型                                       |
| -------- | ---------------------------------------- |
| iostream | istream, wistream从流读取数据。ostream, wostream 向流写入数据。iostream，wiostream 读写流。 |
| fstream  | ifstream, wifstream 从文件读取数据。ofstream, wofstream 向文件写入数据。fstream, wfstream 读写文件。 |
| sstream  | istringstream, wistringstream 从string读取数据。ostringstream, wostringstream 向string写入数据。stringstream, wstringstream读写string。 |

**fstream与sstream都继承自iostream**



## IO对象无拷贝或赋值

```c++
ofstream out1, out2;
out1 = out2; //错误的，无赋值
void print(ofstream stream){}
print(out2) //错误，不能拷贝流对象
```

由于不能拷贝流对象，我们就无法将流对象作为形参、返回值类型。*进行IO操作的函数通常以引用方式传递，而且读写一个IO对象会改变其状态，因此传递和返回的引用不能是const的*



## IO对象的状态

| 流的状态          | 描述                                       |
| ------------- | ---------------------------------------- |
| strm::iostate | iostate是一个机器相关的类型，提供了表达条件状态的完整功能         |
| strm::badbit  | 指出流已崩溃。如果badbit置位 s.fail() 返回true        |
| strm::failbit | 用来指出一个IO操作失败了。如果failbit置位，s.fail() 返回true |
| strm::eofbit  | 用来指出流已经达到了文件结束。如果eofbit置位，s.eof() 返回true |
| strm::goodbit | 用来指出流未处于错误状态。此值保证为0？                     |



几个和流相关的方法：

| 方法                | 描述         |
| ----------------- | ---------- |
| s.good()          |            |
| s.clear()         | 所有的条件状态位复位 |
| s.clear(flags)    |            |
| s.setstate(flags) |            |
| s.rdsrate()       |            |



**确定一个流的状态的最好的方法是将它当做一个条件使用**



## 输出缓冲

每个输出流都管理一个缓冲区，用来保存程序写入的数据。

```c++
cout<<"hello world";
```

执行上述代码，文本串可能立即打出来，也可能被操作系统保存在缓冲区，随后再打印。*有了缓冲机制，操作系统就可以将程序的多个输出操作组合成单一的系统级写操作。*



**什么情况下回导致缓冲区刷新：**

* 程序正常结束，即作为main函数的return操作的一部分。
* 缓冲区满时，需要刷新缓冲，而后新的数据才能继续写入缓冲区
* 可以使用`endl`来显示刷新缓冲区
* 在每个输出操作之后，可以使用操作符 `unitbuf` 设置流的内部状态， 来清空缓冲区。默认情况下，对 `cerr` 是设置 `unitbuf`的，因此写到`cerr`中的内容都是立即刷新的。
* 一个输出流可能会关联到另一个流，在这种情况下，当读写被关联的流时，关联的流的缓冲区会被刷新。？？没搞明白




**刷新输出缓冲区：**

```c++
cout<<"hi"<<endl; //像缓冲区插入换行符，然后刷新
cout<<"hi"<<flush; //刷新缓冲区
cout<<"hi"<<ends; //像缓冲区插入空格，然后刷新
```

## 文件输入输出流

将文件与文件流关联起来，操作流就等价于操作文件。



| 对象                     | 描述                                       |
| ---------------------- | ---------------------------------------- |
| fstream fstrm          | 创建一个未绑定的文件流.                             |
| fstream fstrm(s)       | 创建一个fstrm文件流，并打开s文件。默认的文件模式mode依赖于fstream的类型 |
| fstream fstrm(s, mode) | 指定了mode                                  |
| fstrm.open(s)          | 打开名为s的文件，并将文件与文件流绑定                      |
| fstrm.close()          | 关闭与fstrm绑定的文件。                           |
| fstrm.is_open()        | 返回一个bool值，指出与fstrm关联的文件是否成功打开或者尚未关闭      |



**判断文件是否打开，直接判断流对象是个好习惯**

* 当一个fstream对象被销毁，close会自动调用

## 文件模式总结

| 模式     | 描述                 |
| ------ | ------------------ |
| in     | 以读方式打开             |
| out    | 以写方式打开。（会丢弃文件原有数据） |
| app    | 每次写操作前，均定义到文件末尾    |
| ate    | 打开文件后，立即定位到文件末尾    |
| trunc  | 截断文件。 （会丢弃文件原有数据）  |
| binary | 以二进制形式进行IO         |

**这些模式和类时相关的，可以可以通过  fstream::mode** 调用。



## string 流

`string`流可以使`string`像IO流一样。 sstream头文件中定义了三个类型来支持内存IO。istringstream:从string中读取数据， ostringstream: 像string中写数据。

| 类/方法            | 说明                                       |
| --------------- | ---------------------------------------- |
| sstream strm    | strm是一个被绑定的stringstream对象。sstream是头文件sstream中定义的一个类型 |
| sstream strm(s) | strm是一个sstream对象， 保存着string `s`的一个拷贝。    |
| strm.str()      | 返回strm所保存的string的拷贝                      |
| strm.str(s)     | 将`s`拷贝到strm中。                            |








## 到底应该怎么理解流？

