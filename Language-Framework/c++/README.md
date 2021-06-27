# sizeof

```c++
char* a = "how are you";
sizeof(a); // 12, 字符串的长度加+1, '\0'

```



# 初始化

https://en.cppreference.com/w/cpp/language/initialization



# C++ 命名空间

> :: 叫做 命名空间操作符
>
> std::cin 说明，要使用命名空间 std 中的名字 cin，即：使用命名空间中的成员

```c++
#include <iostream>
using std::cin; // 这样过后，就可以在此源代码中 使用 cin了

int main(){
  int i;
  cin >> i; //cin 这个名字已经认得了
}
```



## 命名空间

> 命名空间的存在是为了解决 变量名污染的问题，命名空间 分割了全局命名空间，其中每个命名空间是一个作用域。命名空间控制着旗下的变量名。

```c++
namespace first_namespace{
  /*声明与定义都可以写在这里。*/
} //命名空间结束后不需要分号
```



**命名空间是一个作用域：**

* 命名空间内的名字要唯一
* 命名空间内的成员可以直接访问此 命名空间内的名字，也可被内嵌作用域的任何单位访问
* 位于该命名空间外的代码，必须指定 命名空间，才可访问此命名空间内的成员



**命名空间可以是不连续的：**

```c++
namespace nsp{
  // 一些声明
}
// 可能是定义了一个名为 nsp 的新命名空间，也可能是对已有的 nsp 命名空间做了补充。
```

> 命名空间的不连续性 可以方便我们 将 函数的声明和定义放在不同的文件中，头文件，源 文件。

​	

**如何使用 命名空间内的成员：**

```c++
#include <iostream>

// 第一种
using namespace std;
cin ...;
cout ...;

// 第二种
using std::cin;
cin ...;

// 第三种
std::cin ...;
```



## 类型别名

```c++
typedef double wages; //wages 是 double 的同义词

typedef wages base, *p; // base 是 double 的同义词，p 是 double* 的同义词
//上面 typedef wages *p 这个地方可能有点怪异，但是如果这么想，*p 与 double 同义，那么 p 与啥同义？

using SI = Sales_item; //SI 是  Sales_item 的同义词
```





## 几点注意事项

* 头文件中不要使用 `using` 声明，以防一些命名冲突。 



# C++  命名别名

在C++中，命名别名有三种方式

* `#define`
* `typedef`
* `using`



## `#define`



## typedef

```c++
// 之后用 ulong 的时候，编译器就会知道是在用 unsigned long
typedef unsigned long ulong; 
```

```c++
// 首先看 fp1 的声明，fp1 指向一个 返回 void * 的函数
// 不加 typedef 表明声明的是 对象，加了 typedef 说明声明的是类型
typedef void * (*fp1)(int);

// b 是 fp1 类型中的一个，说明 b 也是指向一个返回 void* 的函数
fp1 b;
```



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



# C++ 中的右值与左值

## 右值与左值

* 右值：**只能出现** 在 等式 右边的值。(即将被销毁的值(临时变量),  std::move 之后的值, 字面值常量)
* 左值：**可以出现** 在 等式左边的值。

## copy vs  move

```c++
class Foo{
  char *buffer;
}

Foo obj1, obj2;

// copy: obj1 分配空间, 然后 obj2 数据复制过去
// move: obj1.buffer = obj2.buffer; obj2.buffer = nullptr; 如果 obj2是右值的时候, "=" 执行的就是 move
obj1 = obj2; 

```




## 右值引用 与 左值引用

* 左值引用：**左值的引用**

```c++
int i = 0;
int &j = i;
```

上面代码中， `i` 是一个左值，因为 它既可以在等式的右边，也可以在等式的左边。`int &j=i;` 正确的表述应为：获得 `i` 的 左值引用。



* 右值 **引用**：**首先也是引用**（右值的引用，只能**绑定到将要销毁的对象**。即：右值引用只能绑定到 右值上。）
  * 临时对象的内存是分配在调用栈上的(并且为常量对象), 一般情况下是由编译器插入一条语句将其析构, 右值引用会使得 插入析构函数的位置 后移. 
  * (**右值引用只是延时了临时对象的析构函数调用时间**)。
  * 右值引用常用在移动构造函数中

```c++
int i = 10;

// 这种用法没有现实意义, 因为右值引用, 会创建临时对象 (调用一次构造函数), 语句过后, 临时对象不会销毁.
// 如果使用 int j, 也是只会调用一次 构造函数 (j的地址压到函数里, 然后在j的地址上构造)
int &&j = i*100; 

// 常用在 移动构造函数的形参 和 普通函数的形参上.
/*
	移动构造函数的形参 (主要是这个作用)
	普通函数的形参: 由于左值引用不能用在右值上, 所以如果对于右值 不想值传递的话, 那就用 右值引用吧.
	返回一个右值引用的理解:
		也是为了正确 调用 接收对象的 移动构造函数.
		如果返回的是个左值的话, 调用的就是赋值构造函数
		如果返回的是个右值的话, 调用的就是移动构造函数
*/
```

对于一个有返回值的表达式来说，如果这个表达式的返回值没有对象来接收，他会创建一个临时对象，**这个对象在语句结束后就会被销毁**。 所以 `i*100` 返回一个 右值。 所以 `int &&j = i*100;` 就是 `j` 获得了 `i*100` 的右值引用。



```c++
int i = 42;
int &r = i;
int &&rr = i; // 错误：i 是左值
int &r2 = i*42; // 错误，i*42 是右值
const int & r3 = i*42; // 正确，可以将 const 引用绑定到右值上, 因为临时变量(右值)是 const 的?
int &&rr2 = i*42; // 正确 ， i*42 是右值
int && obj2 = func_return_obj(); // 右值引用可以使得 析构函数到 obj2 销毁的时候再执行.
```



```c++
// 右值形参的陷阱
void foo(std::string&& str) {
  std::string s1 = str; // 这里要注意, 虽然形参是右值, 但是进入函数体之后 会被作为左值处理. 
  std::string s2 = std::move(str); // 如果想要恢复右值功能, 还是需要显式move一下.
}
```





## move 函数

虽然不能将 右值引用直接绑定到 一个 左值上，但是我们可以显示的将一个 左值转换成对应的 右值引用类型。

* move 之后，可以对源对象进行 销毁，赋值。**但是不能直接用**
* move **仅仅是做了强制类型转换**, 将值转成右值!!!!!!!
* 一个对象被 move 之后进入一个神奇的状态: 
  * 只能重新赋值 或者 调用析构函数, 其它的操作是违反编程规范的.
* move仅仅是强制类型转换, move之后的行为是由 **move operator=** 决定的.
* **函数的返回值 不需要 move**

```c++
int ii = 111;
// 虽然我们有一个左值，但是我们希望像右值一样对待它

// 实现方式是将数据 移动构造到一个临时对象上??????
int && j = std::move(ii); // move 之后就变成了一个 右值, 就可以用  右值引用 引用之
// 上面 这个 操作之后，除了对 ii 赋值 或者 销毁 它， 我们将不再用它。
```



## 移动构造函数 和 移动赋值运算符

**移动构造函数，移动赋值运算符需要做的事情**

* 完成 **资源移动**  (**利用右值引用 延迟资源 释放时间的 性质**)
* 确保移动后的源对象处于 **销毁无害的** 状态
  * 即：清理干净 源对象

```c++
/* 移动构造函数的两段式写法
1. 先move过来
2. 将之前的 对象的进行 reset
*/
StrVec::StrVec(StrVec&& s) noexcept // 移动操作不应该抛出任何异常
  : elements(s.elements), first_free(s.first_free), cap(s.cap){
    // 顶 s 进入这样的状态--对其运行析构函数是安全的
    s.elements=s.first_free=s.cap=nullptr;
  }
```

```c++
/* 移动赋值运算符, 四段式写法
1. 判断是不是自己给自己赋值
2. 将自己给清空
3. move过来
4. 将之前的对象 reset
*/

StrVec &StrVec::operator=(StrVec &&rhs) noexcept
  {
    if (this!=&rhs){
      free(); //释放已有元素
      elemets = rhs.elements; // 从 rhs 中接管资源。
      ...;
      ...;
      // 将 rhs 处于 析构状态。
      rhs.elements=rhs.first_free=rhs.cap=nullptr;
    }
  }
```



**注意**

* 如果没有定义右值构造函数，那么对于 `move` ，编译器将执行 `copy`



## 完美转发

* `T&&` 被称之为 forward reference.
  * 传给 foo 一个 左值或左值引用: bar会收到  左值引用
  * 传给 foo 一个 右值引用: bar接收到右值引用.

```c++
template <typename T>
void foo(T&& value) {
  bar(std::forward<T>(value));
}
```





# C++ 类型转换

[https://stackoverflow.com/questions/332030/when-should-static-cast-dynamic-cast-const-cast-and-reinterpret-cast-be-used](https://stackoverflow.com/questions/332030/when-should-static-cast-dynamic-cast-const-cast-and-reinterpret-cast-be-used)



## 隐式类型转换

> 编译器偷偷帮我们做的类型转换

**下面的情况中，编译器会自动地转换运算对象的类型**

* 在多数表达式中： 比 `int` 类型小的整形值首先提升为 较大的整数类型。
* 在条件中：非布尔值 转换成 布尔值
* 初始化过程中，初始值转换成变量的类型 **(通过构造函数)**
* 赋值语句中，右侧运算对象转换成左侧运算对象 (**通过赋值构造函数**)
* 如果 算术运算符 或 关系运算符的运算对象 有多种类型， 需要转换成同一种类型。
* **函数调用时也会发生类型转换**

```c++
 int i = 1;
 double j = 3;
 int b = 2;
 double d = i / b; // 结果是 0， i，b 都是整形，整形除的结果也是整数是 0, 0再转成 double，还是 0
```



## 强制类型转换



**`cast_name<type>(expression)`**

* `type` 是转换的目标类型。如果 `type` 是引用类型, 则结果是**左值**。
* `expression` 是要转换的值。
* `cast-name` 指定了执行的是哪种转换。有以下选择
  * `static_cast`
  * `dynamic_cast`
  * `const_cast`
  * `reinterpret_cast`

**`static_cast`** 

> **任何具有明确定义**  的类型转换， 只要不包含底层 const，都可以使用 static_cast。
>
> 常用来处理显示的进行 隐式转化 (int->float, int* ->void*)
>
> (type) value; 之前这种形式的也可以使用 static_cast
>
> 在 const_cast 部分解释 底层 const



**`const_cast`**

> 只能改变 运算对象的底层 const, 可以加上const, 可以去掉const

```c++
const char * pc; // pc 是底层 const 的，因为它指向了 常量
char *p = const_cast<char*>(pc); // 正确，但是通过 p 写值是未定义的行为
char *ps = static_cast<char*>(pc); // 错误，因为 pc 是底层 const 的。

const int i=0;
int j = static_cast<double>(i); //正确，i 不是底层 const， i 是自身 const
```



```c++
const char *cp;

static_cast<char*>(cp); // 错误：static_cast 不能转换掉 底层 const 性质。
static_cast<string>(cp); // 正确，字符串字面值 转换成 string 类型
const_cast<string>(cp); // 错误， const_cast 只改变 底层常量属性！！！
```



**`reinterpret_cast`**

> 通常为 运算对象的位模式提供较低 层次上的重新解释。
>
> 不能移出掉 const 属性, 只有 const_cast 可行

```c++
int *ip;
char *cp = reinterpret_cast<char*>(ip); //换个角度看 ip ，但是记住：pc 的真实对象依旧是一个 int*
```



**`dynamic_cast`**

> 仅仅用来处理多态 (类别要么生命虚函数, 要么继承了虚函数)
>
> 用于 将基类的 **指针** 或 **引用 ** 安全的转换成 派生类的 **指针** 或 **引用** 
>
> 可以向上转型, 可以向下转型, 
>
> * 如果指针转型失败, 就会返回 nullptr.
> * 如果应用转型失败, 会返回` std::bad_cast`

```c++
dynamic_cast<type*>(e);
dynamic_cast<type&>(e);
dynamic_cast<type&&>(e);
```



**需要转型的类型**

* 类型转换 (`int->float, float->int, int->bool`)
  * `static_cast<target_type>()`
* 有继承层次的类型转换
  * 向父类转型: 这个没有必要使用任何特别指示
  * 像子类转型: 
    * `static_cast<>()` : 可以, 但是不会检查类型转化的正确性
    * `dynamic_cast<>()`: 
* 常量类型转换:
  * `const_cast<>()` : 接触const, 也可以加上 const
* 随意解释



# C++循环



## for循环

```c++
for (int i=0; i<10; ++i){
  do something!
}

1. init
2. judgement
3. do something
4. expression --> 2
5. finish.
```



# 预处理

* `gcc -E a.c` : 预编译指令
* `#inlcude` ：世纪就是将文件复制粘贴到该代码位置
* `#include "file"` vs `#inlucde <file2>`: 
  * `trick`: `gcc a.c --verbose` 可以看到更多的日志
  * `#include <file>`: 会去系统的目录中找。如果想将一个目录添加到的系统目录中，可以使用 `gcc -I../` 这种方式添加。看着 `-I./` 语法有点诡异。如果需要将多个目录添加到系统搜索目录，那就多几个 `-I`; `gcc -Ipath1 -Ipath2`



# 编译与链接

* 编译：
  * `gcc -S a.c`，  得到 `a.s` , 汇编代码。
  * `gcc -c a.c`, 得到 `a.o` , 二进制机器码. 可以使用 `objdump -d a.o` 查看机器码
* 链接：
  * `gcc a.o other.o -static`: `-static` 表示将其它的标准库都静态链接到一起。
  * `gcc -lpthread` : `-l` 接的是动态库。
  * `gcc -rdynamic`  采用动态链接方式链接。
* `gcc -D__some_macro__` 用来指定宏



* `make -nB` 只打印编译日志，不进行真实编译



# C语言基础

* 一切皆可取地址

```c
void printptr(void *p) {
  printf("p = %p, *p = %016lx\n", p, *(long *)p);
}

int x;

int main(int argc, char* argv[]) {
  printptr(main);
  printptr(&main);
  printptr(&x);
  printptr(&argc);
  printptr(argv);
  printptr(&argv);
  printptr(argv[0]);
}
```





# 其它

* 字符串常量是直接存储在`二进制`文件中的。



# 如何阅读源代码

* 下载下来
* `tree` 看一下目录结构
* 从 `main` 开始看代码
* 如何读汇编代码：
  * 将内存画出来
  * 将寄存器画出来
  * 然后一步步读 

# ABI vs API

* `ABI`： 约定的是二进制文件格式
* `API`： 程序源代码中的规范



# 链接与加载

* `size a.out` ： 看 `a.out` 的大小
* `nm a.out`: 打印符号与地址
* `objdump -d a.out`: 仅将 text 部分反汇编
  * `objdump -D a.out`: 将所有部分反汇编
* `readelf -a a.out` ：查看重定位信息
