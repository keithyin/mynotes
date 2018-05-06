# swig

**swig是啥**

swig 是一个 `interface complier` ，可以通过它将 c/c++ 代码与脚本语言（`Python，Perl，Ruby`）结合起来。



**swig是怎么工作的**

它通过 c/c++ 的头文件来生成对应脚本文件的 `wrapper code`



## 例子

python 调用 c 代码

```c++
/* File : example.c */
 
 #include <time.h>
 double My_variable = 3.0;
 
 int fact(int n) {
     if (n <= 1) return 1;
     else return n*fact(n-1);
 }
 
 int my_mod(int x, int y) {
     return (x%y);
 }
 	
 char *get_time()
 {
     time_t ltime;
     time(&ltime);
     return ctime(&ltime);
 }
```

```c++
//接口文件，给 swig 看的
/* example.i */
 %module example
 %{
 /* Put header files here or function declarations like below */
 extern double My_variable;
 extern int fact(int n);
 extern int my_mod(int x, int y);
 extern char *get_time();
 %}
 
 extern double My_variable;
 extern int fact(int n);
 extern int my_mod(int x, int y);
 extern char *get_time();
```

```shell
# 创建动态链接库
# 会生成一个 example.py 文件和 一个 example_wrap.c 文件
swig -python example.i 
g++ -c -fpic example.c example_wrap.c -I/usr/local/include/python3.5
g++ -shared example.o example_wrap.o -o _example.so
```



```c++
// swig is lazy
%module example
  %{
  /* Includes the header in the wrapper code */
  #include "header.h"
  %}

/* Parse the header file to generate wrappers */
%include "header.h"
```

