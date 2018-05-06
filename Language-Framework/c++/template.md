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
  void ::Blob<T>::push_back(T &&t){
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

> 类的方法是一个 模板

