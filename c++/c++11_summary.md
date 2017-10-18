# C++11 特性总结

* `lambda` 表达式

> 当我们编写了一个 `lambda` 后，编译器将该 表达式 翻译成一个 未命名类的未命名对象。



* `function` 类 ，在 `functional` 头文件中

> C++ 中的可调用对象：
>
> * 函数
> * 函数指针
> * lambda 表达式
> * 重载了 函数调用运算符的 类的对象
> * bind 创建的对象
>
> 和其它对象一样，可调用对象也有类型。例如：**每个 lambda 有它自己唯一的（未命名）类类型**
>
> 函数与函数指针的类型 由： **返回值类型**和**实参类型**决定。
>
> 两个不同的可调用对象 却可能共享同一种调用形式。

```c++
// int(int, int)

//对应的函数指针类型是 int (*) (int, int)
int add(int i, int j){
  return i+j;
}

auto mod = [](int i, int j){
  return i%j;
}

// 函数对象类
struct divide{
  int operator() (int denominator, divisor){
    return denominator / divisor;
  }
}

```

> 上面这些函数对象 的类型各不相同， 但是它们共享一种调用形式：`int(int, int)`

```c++
#include <functional>
using namespace std;
function<int(int, int)> f; //f是用来存储可调用对象的 空 function，<int(int,int)> 指明了可调用对象的调用形式。

function<T> f(nullptr); // 显示创建一个空 function

function<T> f(obj) ; //f 中保存可调用对象 obj 的副本

f; // f 可以作为条件，当 f 中有可调用对象时 为真， 否则为 假

// demo
function<int(int,int)> f = add;
f(3, 4);
```

```c++
// 重载的函数 与 function， 对于重载的函数，要多费些代码了
int add (int, int){}
int add (float, float){}

int (*fp1)(int, int) = add; 
// 这样来消除 二义性， 为啥不根据接收的 对象来消除 二义性。
function<int(int, int)> f = fp1;
```

