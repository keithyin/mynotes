# C++11 特性总结

* [lambda](#lambda)
* [function class](#function)
* [enum class](#enum class)

## lambda

> 当我们编写了一个 `lambda` 后，编译器将该 表达式 翻译成一个 未命名类的未命名对象。



## function

头文件 `functional` 中

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


```c++
int add(int a, int b) {
  return a + b;
}

int main() {
  // 定义了一个函数指针（未初始化）。该指针可以指向一个 int(int, int)函数
  int (*pf) (int, int);

  pf = add; // 当使用函数名作为 值时，会自动转成指针
  pf = &add; // 该语句和上面的等价
  
  pf(2, 3);
  (*pv)(2, 3); // 这两个调用也是等价的
  
  int (*pf2) (int, int) = add; // 声明并初始化
}


```


函数类型
```c++
// Func 和 Func2 是函数类型
typedef bool Func(const string&, const string&);
typedef decltype(lengthCompare) Func2; // equivalent type

// FuncP 和 FuncP2是函数指针类型。注意和定义函数指针的区别！！decltype 返回的是函数类型，而不是函数指针类型！
typedef bool(*FuncP)(const string&, const string&);
typedef decltype(lengthCompare) *FuncP2; // equivalent type


// 虽然 一个用函数类型，一个用函数指针类型，但是下面的声明是等价的。第一种情况，编译器会自动将 Func 转成指针类型
void useBigger(const string&, const string&, Func);
void useBigger(const string&, const string&, FuncP2);


// 返回函数指针的函数
using F = int(int*, int); // F is a function type, not a pointer
using PF = int(*)(int*, int); // PF is a pointer type

PF f1(int); // ok: PF is a pointer to function; f1 returns a pointer to function. f1是一个函数，int为形参，返回一个 函数指针
F f1(int); // error: F is a function type; f1 can’t return a function。这里要和函数类型作为 函数参数区分开。奇奇怪怪
F *f1(int); // ok: explicitly specify that the return type is a pointer to function。f1是一个函数，int为形参，返回一个指针，该指针指向一个函数

int (*f1(int))(int*, int); // f1是一个函数，int为形参，返回一个指针，该指针指向一个函数


auto f1(int) -> int (*)(int*, int); //使用 trailing return 声明一个变量


// 一个map，保存函数指针.  int ()(int, int) 函数类型。int (*) (int, int) 函数指针。
map<string, int(*)(int,int)> binops;
```

std::function保存任何callable对象
```c++
// ordinary function
int add(int i, int j) { return i + j; }
// lambda, which generates an unnamed function-object class
auto mod = [](int i, int j) { return i % j; };
// function-object class
struct divide {
  int operator()(int denominator, int divisor) {
    return denominator / divisor;
  }
};


function<int(int, int)> f1 = add; // function pointer
function<int(int, int)> f2 = divide(); // object of a function-object class。这个是一定要注意的！！！传一个可调用对象
function<int(int, int)> f3 = [](int i, int j) // lambda
  { return i * j; };
cout << f1(4,2) << endl; // prints 6
cout << f2(4,2) << endl; // prints 2
cout << f3(4,2) << endl; // prints 8
```


## enum class

[http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf)

enum 的问题：

* 容易隐式转换成 int 值
* 。。。
* 。。。

`enum class` ：

* 强类型的，不会隐式转换了。



## constexpr (常量表达式)

**常量表达式**

* 值不会改变
* 编译过程就能得到结果

```c++
const int max_files = 20; // max_files 是常量表达式
const int limit = max_files + 1; // limit 是常量表达式
int staff_size = 27; // staff_size不是常量表达式，staff_size 的值可以被改变
const int sz = get_size() ; // sz 不是常量表达式，因为，运行中才能知道结果
```



* constexpr  变量，C++11 允许将变量声明为 `constexpr` 类型，以便由编译器来验证变量的值是不是一个常量表达式。
  * 声明为 `constexpr` 的变量一定是一个常量，而且必须用常量的表达式初始化
  * 一般来说，如果你认定 变量是一个 常量表达式，那就把它声名为 `constexpr`

```c++
constexpr int mf = 20; // 20是常量表达式
constexpr int limit = mf + 1; // mf + 1 是常量表达式

// 只有当 size 是一个 constexpr 函数时，才是正确的声名语句
constexpr int sz = size(); 
```



* `constexpr` 函数
  * 声明为 `constexpr` 函数会被编译器搞成内联的





## 类型推断

* `auto`
  * 顶层常量会被忽略
    * 底层常量不会忽略，常量对象取地址是一个底层 `const`
    * 如果 `auto &a = var;` 的话，顶层常量不会被忽略。为了保证 var 的常量性
  * 引用会被忽略

```c++
int i=0, &r=i;
auto a = r; // r是个引用，但是 a 不是引用类型

const int ci = i, &cr = ci;
auto b = ci; // b 不是常量，因为顶层常量会被忽略
auto c = cr; // c是个整数，不是常量，也不是引用
auto e = &ci; // e 是一个指向证书常量的指针，常量

auto &g = ci; // g是个整形常量引用
```



* `decltype`
  * 顶层常量和引用都不会忽略

```c++
const int ci=0, &cj = ci;
decltype(ci) x = 0; // x 是 const int
decltype(cj) y = x; // y 是 const int &
decltype(cj) x;     // 错误，引用必须要初始化
```



