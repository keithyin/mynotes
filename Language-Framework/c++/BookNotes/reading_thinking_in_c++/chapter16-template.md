# chapter 16

**模板介绍**

> 一个类型可以给 多种类型的数据用



**语法**

* `template` 这个关键字会告诉编译器，**随后的类定义 将操作一个或更多未指明的类型。**
* 当用这个模板产生实际类代码是，必须指定这些类型以使 编译器能够替换他们。



**模板编译：**

* 为了生成一个实例版本，编译器需要 **掌握函数模板或类成员函数模板的定义** 。因此，与非模板代码不同，模板的头文件中 通常既包含 声明，也包含 模板定义。
* 这个对于 控制实例化 来说是个例外： **如果使用 控制实例化， 头文件可以不包含 模板的定义**



## 函数模板



```c++
template <typename T>
  bool compare(const T& a1, const T& a2){
    // do something...
  }
```





## 类模板

* 在引用模板类名的地方，必须伴有该**模板的 参数列表**
* **模板参数列表** `template <class T1, class T2>` 
* 向模板传参：`<int, float>`



```c++
template <class T>
class Demo{
  T get_val();
  void set_val(T val){
  	// do something..  
  }
  
}

//类 非内联函数的定义
template <class T>
T Demo<T>::get_val(){
  // do something
}

int main(){
  Demo<int> demo; //模板实例化
  return 0;
}
```



## 非类型模板参数

**即：模板参数 不是类型！！**

* 非类型模板参数 表示一个 值，而不是一个类型
* 当模板被实例化时， 非类型参数被 用户提供 或由编译器推断出。 **这些值必须是 常量表达式**

```c++
template <unsigned N, unsigned M>
  bool compare(const char (&p1)[N], const char (&p2)[M]){
    // do something.
  }
```



## inline 模板

```c++
template <typename T>
  inline T min(T a[]){
    // do something.
  }
```



## 控制实例化

当模板被使用时才会被 实例化， 这一特性意味着，相同的实例 可能存在多个对象文件中(`.o` 文件)。 我们可以通过 显示实例化 来避免这个开销。

* 疑问： 那之前 C++ 是怎么解决 重定义问题的？？？？



**用法：**

* `template_demo.h` 中 写 模板的 声明或定义

  * 头文件中，可以写模板的 声明和定义， 也可以只写 声明
* `template_demo.cc` 中 写 模板的定义 和 实例化语句, `#include "template_demo.h"`
  * 如果 `.h` 中模板定义的话， 这里就不用写 定义了。 就直接 写 实例化语句就可以了
  * 如果 `.h` 中没有模板定义的话， 就先 写定义， 然后写 实例化语句
* 对于其它要用 此 模板的 代码 
  * 包含其头文件 `#include "template_demo.h"` (此头文件有无 模板定义都可) 
  * 然后 想用 模板的 那个实例就  `extern template declaration` 
  * 这句话 告诉编译器， **不要在这里 实例化一个 模板实例**，其它地方已经 有这个实例化了， 留给链接搞定剩余的问题就可以了。


```c++
extern template declaration; //实例化 声明语句，不会实例化模板， 意思是模板已经在其它地方被实例化
template declaration; // 实例化定义语句， 实例化 模板。
```



```c++
// template_demo.h
template <typename T>
  bool compare (T a, T b);
```

```c++
// template_demo.cc
#include "template_demo.h"
template <typename T>
  bool compare (T a, T b){
    // do something.
  }

template bool compare(int a, int b); // 模板实例化语句
```



```c++
// main.cc
#include "template_demo.h"
extern template bool compare(int a, int b); // 实例化声明， 说明模板在其他地方已经实例化。

int main(){
  int i=2;
  int j=3;
  bool res = compate<int>(i, j);
  return 0;
}
```





## 模板特例化

**为什么**

* 编写单一模板， 使得对任何 可能的模板实参 都是最适合的，都能实例化， 这并不是总能办到



**所以模板特例化就是， 给模板特例化 一个 实例。**

```c++
template <typename T> // 通用模板定义
  bool compare(const T& a1, const T& a2){
    // do something.
  }

// 特例化, 这是上面模板的一个特例
template<> bool compare(const int* a1, const int* a2){
  // do something.
}
```

