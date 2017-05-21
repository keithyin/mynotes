# c++ string

在介绍c++ string之前，先看一下古老的`char*`。

```c
char name[] = "earth"; // 在栈上创建了创建了一个size为6的数组，数组里的值可以被改变。
char* pname = "earth"; //创建了一个指针，指向字符串常量。
```

## c++ 中的string

标准库类型`string` 表示可变长的字符序列，使用`string`类型必须首先包含`string`头文件。

### 定义和初始化string对象

```c
string s1; //默认初始化，s1是个空串
string s2(s1); // s2是s1的副本
string s2 = s1; //等价于s2(s1)
string s3("value"); // s3是字面值"value"的副本，除了字面值最后的那个空字符外
string s3 = "value"; // 等价于s3("value")
string s4(n, 'c'); // 把s4初始化为有连续n个字符c组成的串
```

### string对象上的操作

```c
os<<s; //将s写到输出流os中，返回os
is>>s; //从is中读取字符串赋给s，字符串以空白分隔，返回is
getline(is, s);//从is中读取一行赋给s，返回is
s.empty();//s为空时，返回true，否则返回false
s.size();//返回s中字符的个数
s[n];//返回s中第n个字符的引用
s1+s2;//返回s1与s2连接后的结果
s1=s2;//用s2的副本代替s1中原来的字符
s1==s2;//判断是否相等，对大小写敏感
s1!=s2;//
<, <=, >, >=;
```



