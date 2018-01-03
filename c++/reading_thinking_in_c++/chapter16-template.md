# chapter 16

**模板介绍**

> 一个类型可以给 多种类型的数据用



**语法**

* `template` 这个关键字会告诉编译器，**随后的类定义 将操作一个或更多未指明的类型。**
* 当用这个模板产生实际类代码是，必须指定这些类型以使 编译器能够替换他们。

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

```c++
extern template declaration; //实例化 声明语句，不会实例化模板， 意思是模板已经在其它地方被实例化
template declaration; // 实例化定义语句， 实例化 模板。
```



```c++
extern template class Blob<string>; // 实例化 声明
template class Blob<string>; // 实例化 定义

// 如何用呢？
Blob<string> blob; // 这里不会再执行 实例化定义操作。
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
template<> bool compare(const T* a1, const T* a2){
  // do something.
}

```

