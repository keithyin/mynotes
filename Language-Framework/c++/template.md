# C++ template



> 模板程序应当尽量减少对函数形参类型的依赖
>
> 模板的声明与定义要放在一起！！！



## 函数模板

```c++
template <typename T> //这个 T 类型，可以在模板声明中当做 类型来用。
int compare(const T &v1, const T&v2){
  if (v1<v2) return -1;
  if (v1>v2) return 1;
  return 0
}
```

> 模板就像是一个公式，可以用来生成对特定类型的函数版本。
>
> 在模板参数列表中，typename 和 class 是一样的。



* `templete <模板参数列表>` 。
* 当使用模板时，我们显式的或隐式的指定模板形参。
  * 显式 ： `compare<int>(1,2)` 
  * 隐式 ： `compare(1,2)` ，模板形参会隐式的推断为 `int`，就会实例化出来一个 版本。



**非类型模板参数：** 

> 模板参数列表中不仅可以放置类型形参，还可以放置非类型参数的形参。一个非类型参数表示一个值 而不是一种类型。我们通过一个特定的 类型名 而非 关键字 typename 或 class 来指定

```c++
template <unsigned N, unsigned M>
  int compare(const char (&p1)[N], const char (&p2)[M]){
    return strcmp(p1, p2)
  }
```



## 类模板

> 编译器不能为 类模板 推断模板参数类型

```c++
template <typename T> class Blob{
  //类里面的属性方法都能用 类型 T
  public:
   void push_back(T &&t);
  private:
}

// 定义函数方法
template<typename T>
  void Blob<T>::push_back(T &&t){
    do_something;
  }


// 实例化类模板
Blob<int> b;
```



**模板参数作用域：**

> 遵循普通的作用域规则

```c++
typedef double A;
template <typename A, typename B> void f(A a, B b){
  A tmp = a; // tmp 的类型为模板参数类型，而不是double
  double B; //错误，重新声明了模板参数 B
}
```





## 默认模板参数

> 跟默认形参似的

```c++
template <typename T, typename F=less<T> >
  int cmmpare(const T &v1, consts T &v2, F f=F()){
    if (f(v1,v2)) return -1;
    if (f(v2,v1)) return 1;
    return 0;
  }

template <class T = int> class Numbers{
  public:
    Numbers(T v = 0): val(v){}
  private:
    T val;
}
```



## 成员模板

> 类的方法是一个 模板， 而类并不是模板类



## 控制实例化

```c++
extern template declaration; // 实例化声明
template declaration;        // 实例化定义

/*例子*/
extern template class Blob<string>; // 在当前文件中并不会对此模板进行实例化，其实例化在其它cpp文件中。链接的时候自会找到
extern template int compare(const int&, const int&);
template int compare(const int&, const int&);
template class Blob<string>; //在当前 cpp 文件中实例化
```

* 显式实例化模板类会实现类的所有成员



## 模板特例化

* 通用模板定义对于特定类型可能是不适合的，这时候就需要 特例化一个模板了
* 函数模板
  * 特例化
* 类模板
  * 特例化
  * 部分特例化，`special specialization`

```c++
// 函数模板特例化
template<> int compare(const char* p1, const char* p2){
  // 特例化代码
}

// 类模板特例化
template<> class Student<People>{
  // 特例化的代码，和通用的部分没有啥关系咯
}

/* 类模板部分特例化， partial specialization
 类的部分特例化依旧是一个模板
*/
template <typename T> struct remove_reference{
  typedef T type;
};
template <typename T> struct remove_reference<T&>{
  typedef T type;
};
template <typename T> struct remove_reference<T&&>{
  typedef T type;
};

// 特例化类模板的某个成员
template<> void Foo<int>::Bar(){
  // 特例化代码
}
```





## Expression Template

[link](http://shoddykid.blogspot.com/2008/07/expression-templates-demystified.html)

* 保存的是表达式对象，`call` 是一个参数
* 表达式对象构建出来一个二叉树。复杂表达式类负责构建二叉树
* Constant 和 Variable 是叶子节点，`call` 是一个参数
