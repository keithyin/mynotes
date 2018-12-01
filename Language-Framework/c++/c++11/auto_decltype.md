# c++11 新兴关键字



## auto

> 类型说明符

```c++
auto v3 = v1+v2; // 会自动推断 v3 的类型
```



## decltype

> 类型指示符

```c++
decltype(f()) sum = x; //会自定推端 f() 返回值的类型。并将其作为 sum 的类型。
```



> 和 auto 的区别，auto 需要计算表示式， decltype 不需要计算表达式。

```c++
// decltype 的结果可以是引用类型
int i=42, *p=&i, &r=i;
decltype(r+0) b; //正确，加法的结果是 int，所以 b 是一个 未初始化的 int

// 解引用操作得到的是 引用。而不是 int
decltype(*p) c; // 错误，c是 int&，所以必须需要初始化。

// 加不加括号的区别
decltype((i)) d; //错误，加了括号，返回 引用
decltype(i) e; // 正确，返回的是 int
```



## constexpr 函数

