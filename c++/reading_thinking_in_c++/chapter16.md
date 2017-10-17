# chapter 16

**模板介绍**

> 一个类型可以给 多种类型的数据用



**语法**

* `template` 这个关键字会告诉编译器，**随后的类定义 将操作一个或更多未指明的类型。**
* 当用这个模板产生实际类代码是，必须指定这些类型以使 编译器能够替换他们。

## 函数模板



## 类模板

* 在引用模板类名的地方，必须伴有该**模板的 参数列表**
* **模板参数列表** `template <class T1, class T2>` 
* 向模板传参：`<int, float>`



```c++
template <class T>
class Demo{
  T get_size();
}

//类 非内联函数的定义
template <class T>
T Demo<T>::get_size(){
  // do something
}

int main(){
  Demo<int> demo; //模板实例化
  return 0;
}
```

