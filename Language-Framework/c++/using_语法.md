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

