# chapter 9

内联函数

> C++ 中，宏的概念是作为内联函数（in-line function）来实现的



## 内联函数

* 类内实现的函数都是内联函数
* 类成员函数：类内不用inline声明，实现的 时候加 inline
* inline 声明的函数。（声明和实现必须放在一起）

```c++
inline void say(){ //声明与实现必须放在一起
  cout<< "hello" <<endl;
}
```

```c++
class Demo{
  void hello();
}
inline void Demo::hello(){
  cout <<"hello world" <<endl;
}
```



* 一般将内联函数定义在头文件里



## 什么样的函数应该声明成内联函数

* 短小精悍的代码
* 长代码不建议



## 神奇的 C++



**C++语言规定：只有在类声明结束后，其中的内联函数才被计算。导致的结果是，类中不需要遵循 先声明再使用这个规则。**

