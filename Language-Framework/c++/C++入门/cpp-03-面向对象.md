# 面向对象

## 类与对象

```c++
class Person{
    int age;
    double height;
  public:
  	void SetAge(int age){this->age = age;}  
};
```

* 访问控制关键字
  * public: 类内，类外都可以访问，子类也可访问
  * protected: 类内访问，子类访问
  * private: 只能在类的内部访问，子类不可访问

* 构造函数与析构函数:(**都没有返回值**)
  * 构造函数：初始化对象
  * 析构函数：对象销毁后的清理工作

```c++
class Person{
  public:
    Person(){}
    
    // 析构函数，无参，无返回值
    ~Person(){}
}
```

* 构造函数的分类
  * 无参构造函数
  * 有参构造函数
  * 拷贝构造函数：一个对象**初始化**另外一个对象

```c++
class Person{
    int age;
  	double height;
  public:
    Person(){}
    Person(int age, double height){
        this->age = age;
        this->height = height;
    }
    
    // 拷贝构造函数
    Person(const Person &person){
        
    }
};

int main(){
    // Person p(); 这个是错误的，会认为是 函数声明
    Person p;
    Person p2(25, 80.0); // 编译器自动调用 构造函数
    Person p3 = {25, 180.0}; // 等号法，对 p3 调用 带参数的构造函数进行初始化
    Person p4 = Person(25, 80.0); // 显示调用法，对 p4 调用 带参数的构造函数进行初始化
}
```



* 对象的初始化 和 对象的赋值不是一个东西

```c++
Person getObj(){
    Person p = Person(25, 100.0);
	return p;
}

int main(){
    Person p1;
    Person p2 = p1;       // 调用拷贝构造函数
    Person p3(p1);        // 调用 拷贝构造函数
                          // 实参初始化形参的时候，会调用 拷贝构造函数
    Person p4 = getObj(); // 接收函数返回值时，
    getObj();             // 不接受也会调用，用来构建匿名对象
    p4 = getObj();        // 调用 = 运算符  
    p2 = p1; //调用 = 运算符
}
```



* 构造函数的初始化列表
  * 如果类中有**常量属性**：要么定义时候初始化，要么初始化列表初始化
  * 如果同时存在 定义初始化 和 初始化列表：将不会执行 定义时候的初始化

```c++
class Leg{
    int num;
  public:
    Leg(num){
        this->num = num;
    }
    Leg(){
        num = 4;
    }
}

class Person{
    Leg leg1; // 构造函数体执行之前，编译器会调用默认构造函数进行对其初始化
    Leg leg2 = Leg(2); //这个就不会在构造函数函数体执行之前进行初始化了
    int age;
    
  public:
    Person(){
        // 如果对象的属性没有定义同时初始化，
        // 这时候会调用属性的默认构造函数进行初始化
        // 然后再执行构造函数的剩下部分
        leg = Leg(); // 这部分是是赋值，而不是初始化
        age = 10;
    }
    
    // 如果不想使用默认构造函数初始化，可以使用初始化列表
    Person(int age, int num_leg1, int num_leg2):age(age),leg1(num_leg1), leg2(num_leg2){
        
    }
    
};
```



* `new delete`
  * `new 与 malloc`： malloc 不会调用构造函数
  * `delete 与 free`：free 不会调用析构函数

```c++
Person *p = new Person;
delete p;

Person *ps = new Person[5];
delete[] ps; // 用来指明 ps指向的是个数组，而不是指向一个简单对象
```



* `static` 关键字
  * 修饰类的属性：属性由类的所有对象共享
  * 修饰类的方法：可以由类名来调用
    * 调用语法：`Person::static_func()` 
  * 修饰函数：
  * 修饰局部变量：
* C++ 中类的属性和方法是分开存储的
  * 静态成员变量：放在全局数据区
  * 普通成员变量：放在结构体里
  * 静态成员函数就没有 `this` 指针

```c++
class Person{
    int a;
  public:
    void print(){
        
    }  
    // 最后的 const 实际是修饰的 this 指针
    void print2() const{ // void print2(const Person* const this)
        
    }
}

// 以上代码会被 c++ 编译器处理成
struct Person{
    int a;
}

void Person_print(Person* const this){
    
}
```



* **运算符重载**
  * 两种实现方法
    * 全局函数方式重载
    * 类成员方法方式重载
  * 返回值需要当左值时  **需要返回引用**
  * `cout  << 10;` 实际上是 `cout.operator<<(10);`！！！！ 
  * `=` 号重载为了支持链式赋值，需要返回引用。
  * 重载 `()` 可以使得对象可以像函数一样调用

```c++
#include <iostream>
using namespace std;

class Complex{
	double real=0.0;
	double img=0.0;
public:
	Complex(double real, double img) : real(real), img(img){

	}
	Complex(){}
	double GetReal()const { return real; }
	double GetImg()const { return img; }
	void Display() const{
		cout << real << "+" << img << "i" << endl;
	}
    // 类成员函数完成重载，左操作数会隐藏
    Complex operator-(const Complex &c2){
        //...
    }
    
};

// 运算符重载实际就是函数重载，将运算符看做函数
Complex operator+(const Complex & c1, const Complex &c2){
	Complex c3 = { c1.GetReal() + c2.GetReal(), c1.GetImg() + c2.GetImg() };
	return c3;
}

// 前置++
Complex& operator++(Complex &c1){}

// 后置 ++, 多一个占位符！！！
Complex& operator++(Complex &c1, int){}

int main(){
	Complex c1(1, 2);
	Complex c2(2, 3);
	Complex c3 = c1 + c2; // 这里会调用 operator+()!!!!
	c3.Display();
	system("pause");
	return 0;
}
```



## 继承

* 继承过程中，不仅继承父类的成员，而且继承父类成员的访问控制，访问控制可以在集成的过程中通过继承访问控制符进行调整。
* 基类中的静态成员，在继承过程中也是所有对象共享的。其余和非静态成员一样
  * 类的静态属性不能在类内设定初始值，只能在类外部
  * `Type ClassName::var = val;` 这样在类外进行初始化

```c++
class Father{
    
};

class Mother{
    
};

class Son: public Father, private Mother{
    
}
```

* **访问控制**
  * 类的方法/属性:
    * public：类内/类外/子类 都可以访问
    * protected: 类内/子类 可访问，类外不可访问
    * private: 类内可访问，类外/子类 不可访问
  * 继承修饰符，表示**继承到子类的属性** 在子类中应该暴露出什么样的访问控制：
    * public：原父类的成员  **在子类中呈现的** 访问属性不变
    * protected：原父类public成员 在子类中变成 protected成员，其余不变
    * private：原父类的 public/protected 成员 在子类中 访问属性变为 private 了

* 类型兼容性原则
  * 子类可以初始化父类对象
  * 父类指针可以指向子类对象



* 继承中的对象内存模型
  * 初始化时，先父类初始化，再子类初始化
  * 析构时，先子类析构，再父类析构



```c++
class Parent{
    int a;
    int b;
    
  public:
    Parent(int a, int b): a(a), b(b){}
    void Print(){
        //...
    }
}

class Child: public Parent{
  public:
    // 初始化列表中对 父类进行初始化，如果不显式初始化，则执行默认初始化（调用默认构造函数）
    Child(int a, int b): Parent(a, b){}
    void Print(){
        Parent::Print();
    }
}
```



* 继承的二义性 与 虚继承

```c++
struct A{
    int a;
}
struct B: public A{
    int b;
}
struct C: public A{
    int c;
}

struct D: public B, public C{
    int d;
}

int main(){
    D d;
    //这儿会出现二义性，因为 B 和C 里面都有a，知道该对哪个赋值，
    // 解决方法：虚继承，保证共同祖先只执行一次构造函数
    // struct B: virtual public A
    // struct C: virtual public A
    // 虚继承只能解决公共祖先的问题
    d.a = 100; 
}
```



## 多态

* 虚函数：基类上对函数加上 `virtual` 修饰符
* 多态三元素
  * 继承
  * 虚函数重写
  * 面向接口编程
* 多态时， **虚析构函数必不可少**

* static binding：在编译过程中就确定了如何执行，在编译阶段就确定了调用哪个函数
* dynamic binding：在运行时候才确定如何执行，像`if, switch`，对于函数调用，在运行过程中才决定调用哪个函数。



**多态实现原理**

* 包含 虚函数的 每个类都有一个虚函数表
* 包含虚函数的基类有一个 `vptr` 指针，子类的 `vptr` 是由继承父类的, 指向虚函数表入口地址
* `vptr` 指针的初始化问题，分步初始化
  * 在执行父类的构造函数的时候，`vptr` 指向父类的虚函数表
  * 父类构造函数执行完毕后，`vptr` 指向子类的虚函数表



**多态调用步骤**

* 调用一个方法， 先判断此方法是不是虚函数
* 如果不是虚函数，则静态绑定
* 如果是虚函数，使用 `vptr` 找函数入口地址



## 纯虚函数抽象类

* 纯虚函数：只有声明，没有定义
* 抽象类：拥有纯虚函数的类为抽象类

```c++
class Person{
  public:
    virtual void Print()=0;
}
```





## 函数模板与类模板

* **类型参数化**
* 模板的声明和定义需要要在同一个文件中，因为编译器生成函数的时候需要知道函数的完整定义
  * 一般写模板代码会有个 `.hpp` 文件，其中就是对模板类/函数的定义
  * 将 `.hpp` include 需要它的地方就可以
* 函数模板
  * 声明：`template <typename T1, typename T2> funcName(T1 t1, T2 t2){}`
  * 调用： `funcName<float, int>(1.0, 3)`
  * 自动类型推导：`funcName(1.0, 3)` 这时候 编译器会 进行自动类型推导
  * 一些特征：
    * **严格的按照类型进行匹配**，不会进行隐式类型转换，普通函数可以进行隐式类型转换
    * 编译器会优先选择函数模板
    * 如果函数模板会产生更合适的匹配，编译器则选择函数模板
    * 使用 `<>`时会 优先使用 函数模板

* 模板原理：**编译过程**
  * **编译器** 在看到 模板函数调用的时候，会根据函数模板生成函数
  * 函数模板的时候：声明和定义要在一起。因为编译器要生成函数的时候需要看到函数模板的定义。



* 类模板
  * 类的声明和定义能不能分开写？

```c++
template <typename T>
class Tmp{
    T t;
}
```



* 继承类模板：
  * 模板类派生普通类：**需要具体化模板类，父类的数据类型得固定下来**
  * 模板类派生模板类：

```c++
template <typename T>
class A{
    T t;
}

class B: public A<int>{
    
}

// 模板类派生模板类
template <typename T>
class C: public A<T>{
    
}
```





* **类中有模板函数**