# chapter 8

**常量**



**创建常量的两种方式：**

* 宏定义
* `const`

```c++
#define BUFSIZE 100
const int BUFFERSIZE = 100;
```



**const 一般放到头文件里**



**C++ 中的 const：**

* 默认为内部连接



```c++
// const 的外部连接用法
extern const int x = 1; // 定义

// another source file
extern const int x; // 声明
```



## const 与 指针

**指向 const 的指针，const 指针**



* 指针的值不允许改变
* 指针指向的值不允许改变
* **C++不允许 将一个 const 对象的`地址` 赋值给一个 非 const ，防止利用指针对 const 对象进行修改。**
* 注意，这里提到的是 地址，而不是值， 值的话，就直接复制了。不用考虑会不会修改原对象的情况。



```c++
const int *u; // u 是个指针，指向 const int

int const *u; // u是个指针，指向 const int

int* const u = new int(1); // u 是个常量指针，指向 int, 因为 u 是个指针常量，所以需要初值

const int* const u = new int(1); // ...

```



**\* 是于标识符结合的，而不是与类型结合**



## const 与 类

* 类内局部 const，（即某个类属性为常数）
* const 对象 （为了支持这个特性，C++ 引入了 const 成员函数，const 对象只能调用 const 成员函数）



**对于第一种情况，解决方案：构造函数初始化列表**

```c++
#include <iostream>
using namespace std;

class Fred{
  private:
  const int size;
  
  public:
  Fred(int sz);
}

// 构造函数初始化列表，对象中 局部常量初始化的地方，当然，非常量也可以在这初始化
Fred::Fred(int sz):size(sz){
  // do something.
};
```



**类中的 static const**

* 编译期的常量，不管类的对象被创建了多少次，都只有一个实例


* 必须在声明的时候定义

```c++
class Demo{
  private:
  static const int size = 10;
};
```



**enum hack**

* 将 不带实例标记的 enum 作为 static const 使用

```c++
class Demo{
  enum {size=100}; //只在类内可见
  int i[size];
};
```



**第二种情况（const 对象 与 const 成员函数）：**

* const 成员函数 可被 const 对象所调用
* 非 const 成员函数 不可被 const 对象所调用
* 构造函数与析构函数都不是 const 成员函数，因为在创建和清理的时候，都会对对象进行修改

```c++
class Demo{
  private:
  int i;
  public:
  Demo(int ii):i(ii){}
  int f() const;
};

int Demo::f() const{
  // do something.
}

// 声明和定义的时候都要加上 const

```



**mutable**

* 在 const 对象中 允许被改变的 成员属性

```c++
class Demo{
  private:
  int i;
  mutable int j; // 即使在 const 成员函数中，也可以被修改
  public:
  Demo(int ii):i(ii){}
  int f() const;
};

int Demo::f() const{
  // do something.
}

// 声明和定义的时候都要加上 const
```





## 总结

**const 可以修饰的有：**

* 对象
* 函数参数
* 函数返回值
* 类成员函数



## 疑问

* 能通过 const 重载吗
* ​

