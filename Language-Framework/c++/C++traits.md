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



