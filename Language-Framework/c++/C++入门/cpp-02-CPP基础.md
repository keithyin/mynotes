# C++基础

* `#pragma once`:  **头文件只被包含一次的意思。**

* 用类型定义变量的三种方法

```c++
struct C1{
  char name[10];
  int age;
};

struct C2{
    char name[10];
    int age;
} c2, c3; //定义结构体的同时定义变量

struct {
    char name[10];
    int age;
} c4, c5; //匿名结构体定义变量

int main(){
    struct C1 c1;
    return 0;
}
```

* 初始化变量的三种方法

```c++
struct C1{
  char name[10];
  int age;
};

struct C2{
    char name[10];
    int age;
} c2={"keith", 25};

struct {
    char name[10];
    int age;
} c4={"keith", 25};

int main(){
    struct C1 c1={"keith", 25};
    return 0;
}
```

* **类定义时候的代码并不是一句一句执行的**： C++编译器将类定义看做一个整体！！！



```c++
// 可以声明并初始化
class Person{
    int age;
    double height = 182.5;    
}
```



* 命名空间

```c++
namespace Space1{
    int a = 10;
    void func1(){
        
    }
}

namespace Space2{
    int a=10;
    void func1(){
        
    }
    namespace NestedSpace{
        int a = 10;
    }
}

void main(){
    int b = Space2::NestedSpace::a;
    using Space2::a ;
    printf("%d", a); //使用 using 开了
}
```



* `const` 修饰符
  * C语言中的 `const` 是个冒牌货，可以通过取地址操作值
  * C++ 中的 `const` 才是真正的 `const`
    * C++ 中将 常量放到符号表(`key-value pair`)中 

```c++
// 前两个是一样的，指向的空间不能修改
const int* b;
int const *b;

// 指针的值不能修改，不能指向另一个地方
int * const b;
```

* `const` 分配内存的时机，在编译的时候分配的内存（栈内存的分配），而不是在运行时



## 引用

* 引用在声明时**必须要初始化**，而且初始化了之后，就不能当其它内存空间的别名了

```c++
int a = 10;
int &b = a; // b 就是 a 的别名了
b = 20; //这时候 a 就是 20 了
```

* 复杂数据类型的引用,  依旧是别名，和原始名字一样效果 

```c++
#include <iostream>

struct Person{
	int age;
	float height;
};

void func1(Person &person){
	Person innerPerson = { 25, 182.0 };
	person = innerPerson;
}

int main(){
	using namespace std;
	Person outter_person = { 18, 150.0 };
	func1(outter_person);
    
    // 输出：25， 182.0
    // 这个和 java 中的行为就不一样了
	cout<<"age: " << outter_person.age << ", height: " << outter_person.height << endl;
	system("pause");
	return 0;
}
```

* 引用的本质
  * 同一块内存空间的别名
  * 是个常量，初始化之后不能当其它内存空间的别名
  * 普通引用也占内存空间，（和指针一样，4个字节）
  * 引用在C++内部实际上是一个 `type *const a`(常量指针)
  * **引用也是一个类型**
* 返回引用（**当返回值需要当做左值时，必须要返回引用**）

```c++
// 返回引用
int & func1(){
    int a = 10;
    // 返回局部变量的引用是有问题的！！！！！
    // 编译器会偷偷的取 a 的地址返回
    return a;
}

int & func3(){
    static int a = 10;
    return a;
}

// 形参为引用
void func2(int &a){
    // do something
}
int main(){
    int a = 10;
    // 编译器会自动给 a 进行取地址操作
    func2(a);
    func3() = 100; // 这样就可以修改 静态变量的值
    
    int b = func3();
    b = 1000; //这样并修改不了 静态变量的值！
    
    int &c = func3();
    c = 10000; // 这样是可以修改 静态变量的值的
}
```



* 指针的引用

```c++
// 指针的引用
void func(int *&a){}
```

* 常量引用
  * 

```c++
int x = 100;
// 用变量初始化常引用
const int &y = x; // 引用的值是常量

int &a = 40;// 普通引用引用一个字面量，字面量是没有内存地址的
```



* 内联函数
  * 不执行函数调用。省去 压栈，跳转，返回 的开销

```c++
// 必须声明和实现写在一起
inline void func(){
    cout<<"hello world."<<endl;
}
```

* 默认参数

```c++
int func(int a, double b, float c=1.0){}
// 函数占位参数，调用时需要传递三个值
void func2(int a, double b, int){}
void func3(int a, double b, int=0){}
```



* 声明函数类型
  * 函数名本身是个 **函数指针** 

```c++
// 这语法号怪异，FuncType 是个函数类型，void (int,int) 的函数类型
typedef void (FuncType) (int a, int b);

// typedef 说明 FuncTypePtr 是个类型，和 变量对比理解
typedef void (*FuncTypePtr) (int a, int b);
int main(){
    FuncType *func = NULL; //只能声明指针类型的
}
```

