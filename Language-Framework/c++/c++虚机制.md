# c++虚机制

`C++`中最玄幻的特性之一就是虚机制，它可以实现动态类型绑定，但是到底是怎么实现的呢？

```c++
class Base{
public:
	virtual void print(){
      cout<<"i am base"<<endl;
	}  
}

class Son: public Base{
public:
  	void print(){
      cout<<"i am Son"<<endl;
	}  
}

int main(){
  Base* b = new Son();
  b->print();
  return 0;
}
//输出 i am Son
```

可以看出，`b->print()` 调用的并不是`Base`类的`print()`方法，而是`Son`类的`print()`方法。这个称为，动态类型绑定。



## 虚函数

为了实现晚捆绑，*C++要求要在基类中声明这个函数时使用virtual关键字*。晚捆绑只对virtual函数起作用，而且只在使用含有virtual函数的基类的地址时发生。

* virtual 关键字只需在声明的时候指定，实现的时候可以不用理会
* 子类中虚函数可以省略virtual 关键字



## C++如何实现晚捆绑

晚捆绑何时发生？所有的工作都由编译器在幕后完成。当告诉编译器要晚捆绑时（通过虚函数告诉），编译器安装必要的晚捆绑机制。

关键字`virtual`告诉编译器它不应该是早捆绑，相反，他应当自动安装对于实现晚捆绑必须的所有机制。



* 典型的编译器会对每个包含虚函数的类创建一个虚表(`VTABLE`)
* 在`VTABLE`中，编译器放置特定类的虚函数的地址。
* 在每个包含虚函数的类中，编译器秘密的放置一个指针，`vpointer`（`VPTR`），指向这个对象的`VTABLE`。
* 当通过基类指针做虚函数调用时，编译器静态的插入  `能取得这个VPTR，并在VTABLE表中查找函地址的` 代码



**无论在什么地方调用虚函数，编译器都会把`VPTR`拿出来走一波。**



# 带有虚函数的基类一定要将析构设置为虚函数

```c++
#include <iostream>

using namespace std;

class base {
   public:
    base() { cout << "Constructing base \n"; }
    virtual ~base() { cout << "Destructing base \n"; }
};

struct SubField {
    SubField() { cout << "Constructing SubField" << endl; }
    ~SubField() { cout << "destruction SubField" << endl; }
};

class derived : public base {
   public:
    derived() { cout << "Constructing derived \n"; }
    // ~derived() { cout << "destructing derived\n"; }

   private:
    SubField sf;
};

int main(void) {
    derived *d = new derived();
    base *b = d;
  	// 如果base的析构不为虚函数，就会导致derived的析构函数不会被调用，就会使得b析构的不够彻底。
    // 将 base 的析构设置为虚函数，一切就可以正常运行了。
    delete b; 
    return 0;
}
```

## 编译时 与 运行时
* 编译时: 代码的 编译的时候
* 运行时: 代码运行的时候
运行时多态?
```
// 当对这个函数进行编译的时候,并不知道 obj 调用的是哪个方法, 只有在运行时,通过传入的对象才知道该调用哪个方法
void CallFunc(BaseClass &obj) {
    obj.SomeVirtualFunc();
}

```

