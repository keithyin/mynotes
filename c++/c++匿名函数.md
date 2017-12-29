# c++ 匿名函数



## 重载函数调用运算符

> numpy 式索引赋值，只不过 [] 要变成 ( )

```c++
#include <iostream>

using namespace std;

//如果一个定义了函数调用运算符，那么该类的对象就是函数对象
class ZhangSB{
public:
    ZhangSB(int age){
        this->age = age;
    }

    int& operator() (){
        return age; 
    }

    int get_age(){
        return this->age;
    }
private:
    int age;
};

int main(){
    ZhangSB zsb(2);
    cout<<"age:" <<zsb.get_age()<<endl;
    zsb() = 10;
    cout<<"age:" <<zsb.get_age()<<endl;
}
```



## c++ 匿名函数

> lambda 是函数对象，函数对象是可被调用的

可调用对象：

* 函数
* 函数指针
* lambda
* 实现 函数调用 方法的 类



**lambda:**

* `capture list:` 捕获列表，是一个 lambda **所在函数中定义的局部变量** 的 列表，通常为空。
  * 一个 lambda 只有在 **其捕获列表中捕获一个它所在 函数中的局部变量**，才能在其函数体内使用。
* `parameter list: `  参数列表
* `return type`： 返回值类型
* `function body` ： 函数体
* **lambda 不能有默认参数**

```c++
[capture_list](parameter_list)->return_type{function_body}
```



```c++
auto f = []{return 42;}  // 可以忽略参数列表和返回类型，但是捕获列表和函数体是需要永远保留的。

//调用, 和普通函数的调用方法一样。
f(); 
```



```c++
// 用引用捕获，会修改 所在函数的局部变量
void test_lambda(){
    int i = 1;
    cout<<"before lambda, the value of i is "<< i <<endl; // 1
    auto lam = [&i]{ i = 2;}; //引用捕获，函数体内部的 i 就是引用。在 lambda 中改变 i 会引起外面的改变。
    lam();
    cout<<"after lambda, the value of i is "<<i<<endl; // 2
}

// 使用 mutable 来更改值捕获的值,默认情况下，捕获的值是不允许更改的，当然更改了捕获的值，也不会修改
// 所在函数的值。
void test_mutable(){
  int v1 = 2;
  auto f = [v1]() mutable{return ++v1;};
}
```



**初探原理：**

当定义一个 `lambda` 的时候，编译器生成一个与 `lambda` 对应的新的（未命名的类类型）。然后 返回的就是这个类类型的对象。这个类对 `operator()` 进行了重载。

