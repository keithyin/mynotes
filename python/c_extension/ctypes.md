# python c语言扩展，使用 ctypes

众所周知，python 语言的执行效率是比较慢的，所以很多计算量大的的操作都是通过 python 调用 c 语言代码实现的。本篇博文将介绍如何 从 python 中访问 c 语言代码。



## 基本流程

* 编写 c 语言代码，并编译成 动态链接库

```shell
gcc -c -fpic example.c
gcc -shared example.o -o _example.so
```



* 在 python 中加载这个动态链接库已供使用

```python
import ctypes
import os
import sys

_file = '_example.so'
_path = os.path.join(sys.path[0], _file)
_mod = ctypes.cdll.LoadLibrary(_path)
```



## 要解决的问题 

* 类型的对应
* 返回多个值的函数怎么搞
* python 对象怎么搞



## 例子

**int 类型** `int ---> ctypes.c_int`

```c
// c语言代码
int add(int x, int y){
  return x+y;
}
```

```python
# python 代码
import ctypes
import os
import sys

_file = '_example.so'
_path = os.path.join(sys.path[0], _file)
_mod = ctypes.cdll.LoadLibrary(_path)

# int add(int, int), 注意一个是 argtypes, 一个是 restype，这也表示了 c 语言只能返回一个值的特点
add = _mod.add
add.argtypes = (ctypes.c_int, ctypes.c_int)
add.restype = ctypes.c_int

# 调用
print (add(3, 10))
```



**double类型** `double ---> ctypes.c_double`



**指针类型** `int* ---> ctype.POINTER(ctypes.c_int)`

```c
// c 语言代码
int divide(int a, int b, int * remainder){
  int quot = a / b;
  *remainder = a % b;
  return quot;
}
```

```python
# load dll 
_divide = _mod.divide
_divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divide.restype = ctypes.c_int
# 这也是返回多个值的一个例子
def divide(x, y):
    rem = ctypes.c_int()
    quot = _divide(x,y, rem)
    return quot, rem.value

```

