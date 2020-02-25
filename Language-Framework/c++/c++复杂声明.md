# C/C++复杂声明



```c++
void (*funcPtr)();
```

上面声明了一个指向函数的指针，当碰到这样一个比较复杂的声明时，最好的方法是从 **中间开始和向外扩展**：

* 从中间开始：即 从变量名开始
* 像外扩展：即 先注意右边最近的项，已右括号结束，再注意左边的项，已左括号结束，再注意右边的项。。。

用上述方法来解析一下第一个声明：

* 往右看：是右括号，结束
* 往左看：funcPtr是个指针
* 往左看：碰到左括号，结束
* 往右看：指针指向一个函数，即：指向函数的指针
* 向左看：函数的返回值类型是 void，即：指向返回值为 void 的函数



再看第二个例子：

```c++
void * (*(*fp1)(int))[10];
```

* 往右看：是右括号，结束
* 往左看：fp1 是个指针
* 往左看：是左括号，结束
* 往右看：fp1 指向一个 参数为 int 的函数（因为函数是 `()` 指示的，碰到了 `)` 下一步就往左看）
* 往左看：函数返回一个指针
* 往左看：是左括号，结束
* 往右看：指针指向一个 数组
* 往左看：数组类型为 void *

即：fp1是一个函数指针，指向的函数 接收 int 为参数，返回一个指针，这个指针指向一个 10 个元素的数组，数组类型为 void *



## 函数类型

```c++
// 首先看 fp1 的声明，fp1 指向一个 返回 void * 的函数
// 不加 typedef 表明声明的是 对象，加了 typedef 说明声明的是类型, fp1 是个 指针类型!!
typedef void * (*fp1)(int);
fp1 b; //声明了一个指向函数的 指针
// fp2 是一个 函数类型.
typedef void * (fp2)(int);

// Func2 是一个函数类型 
typedef decltype(someFunc) Func2;

// Fun3 是一个 [函数] 指针类型. 和 typedef void * (*fp1)(int); 类似
typedef decltype(someFunc) *Func3;

// F 是个函数类型
using F = int(int*, int*);
// FP是个指针类型
using FP = int(*)(int*, int*);

F f1(); // error: 函数是不可能返回一个函数的, 只能返回一个函数指针!!!
F* f2(); // ok!
FP f3(); // ok!
```



```c++
// 以下两个声明等价.
int (*f1(int))(int*, int);
auto f1(int) -> int (*)(int*, int);
```



## lambda 表达式

* lambda 作为形参, 应该怎么操作: 使用 `std::function` 接
* lambda 作为返回值怎么办: 是用 `std::function` 接

## std::function

* 一个类,  函数类!!!, 构建任何可调用对象
* 可调用对象包括
  * 指向函数的指针, 函数, lambda表达式, bind之后的返回值, 重载了 `()` 的对象

```c++
#include <functional>
// <> 里面放的是函数的签名,
function<int(int, int)> f1 = add; // 函数指针
function<int(int, int)> f2 = div(); // 函数对象, 原来 带了括号的是函数对象啊, 怎么和函数调用区分开的呢?
function<int(int, int)> f3(int i, int j) {return i*j;} // lambda表达式
f1(1,2);
f2(1,2);
f3(1,2);
```

