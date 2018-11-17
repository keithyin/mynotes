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

