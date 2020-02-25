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



* 值捕获 **不会更改 lambda 所在函数的局部变量**, 
  * **用 mutable** 将 read-only 的值捕获过来的变量 变得可写, 然后还会保存其状态
  * 不会修改之前的栈上的值, 这个和 引用捕获不一样.

```c++
int main() {
    int a = 1;
    int b = 2;
    auto func = [a, b]() mutable {
        a += 1;
        b += 1;
        std::cout << "lambda a=" << a << ", lambda b=" << b << std::endl;
    };
    func();
    func();
    std::cout << "a=" << a << ", b=" << b << std::endl;
    return 0;
}
/*
lambda a=2, lambda b=3
lambda a=3, lambda b=4
a=1, b=2
*/

```

* 引用捕获, 
  * 修改的状态会被保存
  * 原来栈中的数据也会跟着改变

```c++
#include <functional>
#include <iostream>

std::function<void()> get_lambda(int &a, int &b) {
    auto func = [&a, &b]() {
        a += 1;
        b += 1;
        std::cout << "lambda a=" << a << ", lambda b=" << b << std::endl;
    };
    return func;
}

int main() {
    int a = 1;
    int b = 2;
    auto func = get_lambda(a, b);
    func();
    func();
    std::cout << "a=" << a << ", b=" << b << std::endl;
    return 0;
}
/*
lambda a=2, lambda b=3
lambda a=3, lambda b=4
a=3, b=4
*/
```

* 如果引用的变量, 栈被销毁, 则会导致行为不确定

```c++
#include <functional>
#include <iostream>

std::function<void()> get_lambda(int a, int b) {
    auto func = [&a, &b]() {
        a += 1;
        b += 1;
        std::cout << "lambda a=" << a << ", lambda b=" << b << std::endl;
    };
    return func;
}

int main() {
    int a = 1;
    int b = 2;
    auto func = get_lambda(a, b);
    func();
    func();
    std::cout << "a=" << a << ", b=" << b << std::endl;
    return 0;
}
/*
lambda a=32767, lambda b=-356927439
lambda a=32767, lambda b=-356927439
a=1, b=2
*/
```





**初探原理：**

当定义一个 `lambda` 的时候，编译器生成一个与 `lambda` 对应的新的（未命名的类类型）。然后 返回的就是这个类类型的对象。这个类对 `operator()` 进行了重载。

