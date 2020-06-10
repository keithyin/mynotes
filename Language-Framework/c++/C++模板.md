#  C++ template

* 解决的问题是

  * 在写代码的时候, 不知道要处理的是什么类型,  比如说 写容器的时候, 我们可以往容器里面塞各种各样的类型. 但是在我们编写容器类的时候, 我们并不知道要往里面放什么类型, 只有当我们用容器类的时候, 才知道往里面放什么类型.
  * OOP也是这样, 在我们 **写**一个接口的时候, 并不知道谁会用它, 只有真实在代码中调用的时候, 才知道是谁调用的.

  * 模板: 编译时才知道
  * OOP虚机制: 运行时才知道 (**编译时不知道**)

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



### 非类型模板参数

> 模板参数列表中不仅可以放置类型形参，还可以放置非类型参数的形参。一个非类型参数表示一个值 而不是一种类型。我们通过一个特定的 类型名 而非 关键字 typename 或 class 来指定
>
> 这玩意有啥意义呢 ? 为啥不通过传参? 对于数组是有意义的, 因为数组需要在编译的时候知道其大小.

```c++
template <unsigned N, unsigned M>
  int compare(const char (&p1)[N], const char (&p2)[M]){
    return strcmp(p1, p2)
  }
```

### inline 应该放哪里?

```c++
// 正确!!
template <typename T> inline T min(T a, T b);
```









## 类模板

> 编译器不能为 类模板 推断模板参数类型

```c++
// 模板的定义和实现应该在同一 .hpp 文件中.
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

* 模板会在使用的地方实例化, 这一特性意味着相同的实例可能出现在多个 `.o` 文件中. 在一个大系统中, 这就导致了额外开销
* 控制实例化就是为了解决同一个实例同时在多个 `.o` 文件中的问题
* 有了控制实例化: 模板的声明和实现可以分离到 `.h` 和 `.cpp` 文件中了.
  * 在 模板实现的 `.cpp` 文件中, 使用 `template  declaration`
  * 在使用模板的 `.h / .cpp` 文件中, `extern template declaration`

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



## 模板重载

* 这里考虑的是当一个同名模板的形参分别为 `const T&, T&, T*, const T*` 时, 啥时候该调用谁的问题
* 规则
  * 如果有非模板函数, 优先选非模板函数
  * 如果都是模板函数, 选择更特例化的

```c++

```





## Expression Template

[link](http://shoddykid.blogspot.com/2008/07/expression-templates-demystified.html)

* 保存的是表达式对象，`call` 是一个参数
* 表达式对象构建出来一个二叉树。复杂表达式类负责构建二叉树
* Constant 和 Variable 是叶子节点，`call` 是一个参数



# traits

## `type_traits`

* `#include <type_traits>`
* 作用: 用作类型转换 (不同于 那几个 `cast` 函数)

```c++
remove_reference;
add_const;
add_lrvalue_reference;
remove_pointer;
add_pointer;
make_signed;
make_unsigned;
remove_extent;
remove_all_extents;
```

```c++
// decltype(*beg) 推断出来的类型是个引用类型, 因为*beg 是个引用.
template<typename It>
auto fcn(It beg, It end)->decltype(*beg) {
	return *beg;
}

// 如果不想返回引用怎么办呢? remove_reference就好了.
// ::type 是模板类的一个静态成员
// typename 用来表征 ::type 搞出来的值是个 类型!!!
template<typename It>
auto fcn(It beg, It end)-> 
	typename remove_reference<decltype<*beg>::type {
	return *beg; 
}
```



# 模板参数推断

> 默认情况下, 编译器使用 调用模板函数时 传入的 参数来决定 模板参数. 这个过程被称之为 模板参数推断.
>
>  During template argument deduction, the compiler uses types of the arguments in the call to find the template arguments that generate a version of the function that **best matches** the given call.

推断规则:

* `top-level const` 被忽略
  * `top-level const` : ie: 指针本身是 const, `low-level const` , 指针所指向的对象是 `const`
* `low-level const` 可以被推断出来
* `const 转换` : 非 `low-level const` 的指针或引用,  可以传递给一个 `low-level const` 的函数形参
* `array 或者 function-to-pointer` 转换:  如果函数的形参不是一个引用类型, then the normal pointer conversion will be applied to arguments of array or function type. An array argument will be converted to a pointer to its first element. Similarly, a function argument will be converted to a pointer to the function’s type
  * 注意: `parameter: 形参` ,`argument 实参`
* 以下转换不会执行: 数值类型转换, `derived-to-base`, 用户自定义的转换; 这些都不会被执行.

```c++
template <typename T> T fobj(T, T);
template <typename T> T fref(const T&, const T&);
string s1("a value");
const string s2("another value");
fobj(s1, s2); // calls fobj(string, string); top-level const is ignored

// calls fref(const string&, const string&), 这里因为 `const 转换存在` 所以才合理
// s1 : 非 low-level const 可以传递给 low-level const 函数形参
fref(s1, s2); 

// uses premissible conversion to const on s1
int a[10], b[42];
fobj(a, b); // calls f(int*, int*)

// error: array types don't match, 因为是引用, 所以推断出来的类型应该是 int[10]和int[42]
// 这两个数组类型是不 match 的.
fref(a, b); 

template <typename T> compare(T a, T b);
long a;
compare(a, 1024);// 会翻译成 compare(long ,int) , 所以会报错 
```

* 当将一个函数模板赋值给一个函数指针的时候, 编译器会根据函数指针的签名来推断函数模板参数

```c++
template <typename T> int compare(const T&, const T&);
// pf1 points to the instantiation int compare(const int&, const int&)
int (*pf1)(const int&, const int&) = compare;

// overloaded versions of func; each takes a different function pointer type
void func(int(*)(const string&, const string&));
void func(int(*)(const int&, const int&));
func(compare); // error: which instantiation of compare?因为函数重载导致的 模板参数推断的二义性.
```

* 左值引用与右值引用

```c++
// 模板形参是一个左值引用, 只能传递给它一个左值
template<typename T> void f1(T&);
f1(i);//i 是 int, T 被推断为 int
f1(ci); // ci是 const int, T 被推断为 const int
f1(5); // 错误: 传递给 一个 & 的实参必须是左值.


// 模板形参是 const T&, 啥都能传
template<typename T> void f2(const T&);
f2(i);//i 是 int, T 被推断为 int
f2(ci); // ci是 const int, T 被推断为 int
f2(5); // const T& 可以绑定 右值, T被推断为 int

//模板形参是 T&&
template<typename T> void f3(T&&);
f3(43); // 可以传右值, T为 int
```

* 引用折叠 与 右值引用参数
  * 两个例外规则: 
    * 当一个左值传递给一个 右值的引用形参, 编译器推断模板类型参数为实参的 左值引用类型
    * 引用折叠:
      *  `X& &, X& &&, X&& &` 都会折叠为 `X&` 
      * `X&& &&` 会折叠为 `X&&`

```c++

```



## 尾返回类型

```c++
// a trailing return lets us declare the return type after the parameter list is seen
template <typename It>
auto fcn(It beg, It end) -> decltype(*beg) // 为什么写在后面, 因为*beg是在编译器看到 形参列表时候才知道其存在的.
{
// process the range
return *beg; // return a reference to an element from the range
}
```



 

读  "stl源码剖析" 总结 

# traits指的是什么

traits, 中文翻译"特性", 刚一看到这个词一脸懵逼...  特性?? 和 类的成员变量(有时也会称之为 属性) 是啥关系. 直到看到两个例子之后, 才对 `traits` 这个词有了真正的了解.

目前了解到的两种 `traits` 是

* `iterator_traits` : 在 `<iterator>` 头文件下
* `__type_traits`

# `iterator_traits`

> 在算法中使用迭代器的时候, 很有可能会使用到迭代器所指向的对象的类型. 而我们在编写算法代码的时候, 并不知道会有什么样的迭代器传进来. 要解决这个问题, 我们刻意利用 **函数模板的参数推导功能**

```c++
template <typename I, typename T>
void func_impl(I iter, T) {
	T tmp; // 通过传入一个 迭代器所指的 值, 就能够在编译期这里是什么类型了.
}

template <typename I>
void func(I iter) {
	func_impl(iter, *iter); // 这里传入一个值
}
```

> 现在, 问题来了, 如果我们想要一个有返回值的 `func`, 而且返回值的类型为 迭代器所指值的类型, 那我们应该怎么办呢?

```c++
// 这是一种方法, 但是看起来是不是有点丑呢?  暴露给用户的 api, 第二个参数的目的仅仅是为了 模板参数推导
template <typename I, typename T>
T func(I iter, T){
	T tmp;
  return tmp;
}
```

> 上面的方法不够美观, 暴露出来了一些没有必要的信息, 所以需要一个更加合适的方法. 另外一种方法是:  既然我们已经传入了 一个  iter, 为啥不让这个 iter 来告诉我们 它指向的对象的类型呢? (iter 所指向的对象的类型 就是 iter 的一个 trait.) 有了这个方法之后, 我们的代码就可以写成以下形式了.

```c++
template <typename I>
typename I::value_type func(I iter){ 
  // typename 是告诉编译器 I::value_type 是个类型 而不是一个值
  typename I::value_type tmp;
  return tmp;
}

// 为了可以这么用, 我们的iterator的实现需要加一些东西了
template <typename T>
struct someiter {
	typedef T value_type;  //模板实例化的时候, 这个类型也就知道了.
  typedef T* pointer;
  typedef T& reference;
  // etc
};
```

> 实际应用中, 我们想从 iter 类想获得的可能不仅仅是 其指向的值的类型, 也想得到 指向对象的指针类型, 引用类型, etc. 这些我们可以称之为 iterator 的 traits 

> 关于 iterator_traits 是个啥目前已经知道的差不多了. 但是离我们实现一个 iterator_traits 还差一步. 那就是, 如果 iter 是个原生指针, 是没有 ::value_type 是取不出来东西的, 这该怎么办呢?  这就需要模板的偏特化 出场了.

```c++
// 非原生指针 iter, 会走这个模板
template <typename I>
  struct iterator_traits{
    // typename 的功能依旧是说明 后面带的是类型, 而不是值
  	typedef typename I::iterator_category iterator_category;
    typedef typename I::value_type value_type;
    typedef typename I::difference_type difference_type;
    typedef typename I::pointer pointer;
    typedef typename I::reference reference;
  };

// 偏特化版
template<typename T>
struct iterator_traits<T*>{
  //...
	typedef T* pointer;
  typedef T& reference;
};

template<typename T>
struct iterator_traits<const T*>{
  //...
	typedef const T* pointer;
  typedef const T& reference;
};
```

> 一旦完成 `iterator_traits` 这个模板, 我们的上述代码就可以写成

```c++
// 此时这个I 是原生指针 还是 class 都可以正常用啦.
template <typename I>
typename iterator_traits<I> func(I iter){ 
  // typename 是告诉编译器 I::value_type 是个类型 而不是一个值
  typename I::value_type tmp;
  return tmp;
}
```

> 小总结: 对于 iterator 来说, 目前将其 traits 定义为 其指向的对象的类型相关的一些 东西.



# `__type_traits`

> `__type_traits` 是 **类型** 的一些特性, 比如: 
>
> * `has_trivial_default_constructor`
> * `has_trivial_copy_constructor`
> * `has_trivial_assignment_operator`
> * `has_trivial_destructor`
> * `is_POD_type`
>
> 这些个玩意具体是啥含义, 还不知道.......  但是通过这个可以看出, 我们定义`class`的时候 , 是用这个 `class` 来描述我们所抽象的一个东西, 比如(`class Person; class Executor`) , 而 `traits` 这玩意含义似乎是描述 `class` 的, 比如(是不是 POD 类型啊, 等等)  这可能就是 `元编程` 的 `元` 吧.

知道了以下步入正题

> 我们在复制的时候 如果我们想 根据 **类型的不同的特性** 来调用不同的 函数进行复制. 我们该怎么做呢?
>
> 类比 我们如何根据不同的 类型 来调用不同的函数进行操作.
>
> 答案就是: 重载.
>
> 感觉我们在 class 里面搞一个 `bool is_POD_type` , 然后根据其值 来走 一个函数的 不同逻辑分支也是可以的吧...  是不优雅吗?
>
> 现在看一下如何使用模板重载来实现.

```c++
// 定义一个 true_type, 和 一个 false_type 结构体, 
struct __true_type{}; // 这是一个 true 的类型, 对象就是一个 true 的对象
struct __false_type{};// 和上同

template<typename type>
struct __type_traits{
	typedef __false_type has_trivial_default_constructor;
  //...
  typedef __false_type is_POD_type;
};

// 以下是一堆特化
template<typename type>
struct __type_traits<char>{
	typedef __true_type has_trivial_default_constructor;
  //...
  typedef __true_type is_POD_type;
};

// 通过 is_POD_type 类型,  通过 重载机制? 来进行 函数分发
template<typename T>
void fun_impl(T t, __true_type){
// do something
}
template<typename T>
void func_impl(T t, __false_type){
// do something
}

template<typename T>
void func(T t){
  //typename __type_traits<T>::is_POD_type() 会根据T的traits决定构建 __true_type对象, 还是 __false_type 对象.
	func_impl(t, typename __type_traits<T>::is_POD_type());
}
```



# 总结

`traits` 表示的应该为抽象的抽象. 我们现在编程的抽象为 `class` , `iter` ,... `traits` 表示的则是 `class` 的一些通用性质, 比如上面提到的.  



# 模板重载



