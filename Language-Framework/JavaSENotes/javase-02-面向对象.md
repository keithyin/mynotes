# 面向对象

**Java中的引用传递实际上是 C++ 中的指针传递**

* 所以 JAVA 中 **只有值传递**



面向对象编程的三条主线：

1. 类及类的构成成分：属性  方法 构造器  代码块 内部类
2. 面向对象编程的特征：封装性  继承性  多态性  （抽象性）
3. 其它的关键字：this super package import static final abstract interface .



**规定**

* 一个 `.java` 文件中有且只有一个 `public` 类，且类名和文件名一致



```java
public class Test{
    public static void main(String[] args){
        People p = new People(); // 在堆中分配内存
    }
}
class People{
    
}
```



**成员变量 vs 局部变量**

* **相同点**
  * 遵循变量声明的格式： 数据类型 变量名 = 初始化值
  * 都有作用域
* **不同点**
  * 位置
    * 成员变量：类内，方法外，声明格式：[修饰符] 数据类型 变量名 = 初始化值;
    * 局部变量：方法内
  * 修饰符
    * 成员变量：四个权限修饰符：public private protected 缺省（即不加修饰符）
    * 局部变量：没有修饰符
  * 初始化值
    * 成员变量：如果声明的时候不显式赋值，则会有默认初始化值
    * 局部变量：一定要显示的赋值（没有默认值），**不赋值就使用就会报错**
  * 内存位置不同
    * 成员变量：堆空间中
    * 局部变量：栈空间中

```java
public class Test{
    public static void main(String[] args){
        People p = new People(); // 在堆中分配内存
    }
}
class People{
    int age = 10; // 成员变量
    public void print(){
        int i = 100; // 局部变量
    }
}
```



**方法:(`public void funcname(int i){}`)**

* 四个访问修饰符：public，private，protected，缺省

* 方法重载：通过形参不同（个数不同/类型不同）进行重载，（**不能通过返回值类型重载！！！**）

```java
class Demo{
    // String...args 可变个数的形参，只能放在最后声明，和 数组一样使用
    public void func(int a, String ... args){
        
    }
}
```



**构造器**

* 作用：创建对象
* 不显式定义时，java 提供一个默认的构造器
* 一旦显式定义时，就不提供默认构造器了
* 类的多个构造器之间构成重载

```java
class Person{
    private String name;
    private int age;
	public Person(){} 
}
```



**类对象属性的赋值顺序**

* 属性的默认初始化 --》构造器中的第一句代码 --》 属性的显式初始化 --》构造器的剩余代码
  * 默认初始化：类对象默认初始值为 null
  * 构造器中的第一句代码：如果不是 `super(...)/this(...)` 则插入 `super()`，然后调用
  * 如果第一句是 `super(...)/this(...)`，就直接调用

**this**

* 表示当前对象（**可以用来修饰：属性，方法，构造器**）
  * `this(arg);` 这样来调用构造器
* 表示当前正在创建的对象（在构造器中使用的话可以这么理解）



**代码块（初始化块）**

* 用来对类的属性进行初始化
* 里面可以有打印语句
* 在 构造器的 剩余代码执行之前操作，**显式声明和代码块按照顺序执行**

```java
public class Demo{
    int age;
    String name;
    {
        age = 10;
        name = "keith";
        System.out.Println("hello")
    }
}
```







## 权限修饰符

**权限修饰符，可访问范围：类内，包内，子类，任何地方**

- `private`: 使用 `private` 修饰的 **方法/属性** 只能在类内使用，类外就不能用了。
- `缺省`：类内，包内，子类（不能直接访问，需要通过父类对象访问）？？？？
- `protected`:  类内， 包内，子类(直接访问，而不是通过对象访问)
- `public` : 任何地方



**修饰类的权限只有两个：public，缺省**



## 关键字

* static
* final
* abstract
* interface



**static**

* 在写类的时候，只是一堆代码，只有在 `new` 的时候，才会分配空间存储对象，每个对象会各自拥有属性的副本。
* 可以用来修饰：属性，方法，代码块，内部类。即：除了不能修饰构造器，类的其它组件都可以
* 修饰属性：**属性变为类变量，所有此类的对象共享**
  * 类变量：随着类的加载被加载（分配资源与初始化）
  * 类变量是放在静态域的
  * 可以直接通过 `类.类变量` 方式调用，使用 `对象.类变量` 调用也可以
* 修饰方法：方法变为类方法
  * 随着类的加载而加载，独一份
  * 可以通过 `类.类方法` 调用
  * 静态方法中只能调用 静态属性和静态方法
* 修饰代码块：静态代码块
  * **随着类的加载而加载**，加载一次就不加载了
  * 静态代码块的执行 要 **早于非静态的**
  * 静态代码块中只能操作静态的结构（类属性，类方法）
* 修饰内部类：



**final**

* 可用来修饰 类，属性，方法
* 修饰类：**类不能够被继承**
* 修饰方法：**方法不能被重写**
* 修饰属性：**属性变为常量，不能被修改了**
  * 常量不能使用默认值了。（可以：显示、代码块、构造器 中赋值）
* `static final` ：全局常量





## 内部类

* 一般定义在 类 或者 语句块之内
* 成员内部类：类内，方法外，是外部类的一个成员
  * 修饰符（4个），一般的类只有两个修饰符
  * 可以用 static，final 修饰
  * 可以用 abstract 修饰
* 局部内部类：方法内
* 匿名内部类：实现接口的内部类



```java
public class Test{
    public static void main(String[] args){
        // 静态内部类的创建
        Person.Leg leg = new Person.Leg();
        
        // 非静态内部类, 需要通过 外部类的对象创建
        Person.Head head = new Person().new Bird();
        
        // new 的时候提供接口的实现
        Demo demo = new Demo(){
            public void show(){
                // do something
            }
        }
    }
}

interface Demo{
    void show();
}

class Person{
    String name;
    int age;
    // 和成员属性，成员方法并列，称为成员内部类
    // 可以调用外部类的成员，因为是外部类的成员
    class Head{
        void func(){
            // 调用外部类的属性
            System.out.Println(Person.this.name)
        }
    }
    
    static class Leg{
        
    }
    
    // 放在方法之中，局部内部类
    public void myFunc(){
        // 局部内部类
        class Do{
            
        }
    }
}
```





## 抽象类

* 关键字：`abstract`，`abstract class Person`
* 修饰 类，方法
* 修饰类：抽象类
  * 不可以被实例化（不能被 new）
  * 虽不可被实例化，但是是有构造器的，还可以自定义构造器
  * 抽象类中可以没有抽象方法
* 修饰方法：抽象方法
  * 没有方法体，不包含`{}`
  * 抽象方法 所在的类，一定是抽象类！！类必须要 `abstract` 修饰
* `abstract` 不能修饰 属性，构造器，private，final，static



## 接口

* **抽象方法** 和 **常量值** 定义的集合，即：只有 抽象方法和常量
* `interface`,    `interface InterfaceName{...}`
* 接口没有构造器的，所有的类都有构造器

```java
interface AA{
    // 默认使用 public static final 修饰的
    int I = 12;
    
    // 默认是使用 public abstract 修饰
    void Method(){
        
    }
}
```





## 继承

* Java 只支持单继承，语法`class A extends B`
* 当父类有 私有的 属性/方法 时，子类同样可以获取的到，只是由于权限问题，子类不能直接调用。但是可以通过调用继承过来的父类的 `public 方法` 来访问 `private 方法/属性` 
* 



```java
class A{
    public void print(){
        // ...
    }
}

class B extends A{
    
    // 重写要求返回值类型必须一样，私有的无法重写，也没有意义
    @override
    public void print(){
        // ...
    }
}
```



**super：调用父类中的指定操作**

* 属性，方法，构造器（`super(形参列表)`，调用父类中指定的构造器）
* `super` 的追溯**不仅限于**直接父类
* 在子类构造器中，如果不显式调用 `super(...)` 或者 `this(...)`，默认会在构造器**第一行插入调用父类空参构造器**的代码。java有个超级父类 `object` **！！！！！！**
* 设计一个类时，尽量提供一个空参构造器，因为 java的很多默认行为都会调用空参构造器



**子类对象实例化的全过程**

* 。。。。。。



## 多态

* 通过 继承类，或者实现接口实现
* 面向接口编程
* 多态针对于方法的调用，属性没有多态性



## 关于 String 类

```java
String str1 = "AA";
String str2 = "AA"; // "AA" 在字符串常量池里
String str3 = new String("AA"); // new 出来的在堆空间，里面"AA"还是在常量池里
str1 == str2;      // true， == 判断存放的地址是否一样
str1.equals(str2); // true，equals 由 string 重载，判断内容是否一样
str1 == str3;      // false
str1.equals(str3); // true
```



## object 类

* `equals` 方法，由子类重写此逻辑
* `toString()` 方法，打印对象的引用时，默认会调用此方法



## 包装类

* 针对 **基本数据类型** 提供的 **引用类型**

* boolean：Boolean，byte：Byte，short：Short，int：Integer，long：Long，char：Character，float：Float，double：Double



## 使用 Junit 单元测试

* 右击项目文件夹--》build path--》Add Libaraies--》JUnit--》junit4

```java
import org.junit.Test

class Person{
	@TEST // 右键此函数，junit 运行即可
	public void someFunc(){
		// do something
	}
}
```





## JavaBean

* 什么是 javabean
  * 满足以下条件的 java对象：
    * 类是 public 的
    * 有一个无参的 public 构造器
    * 有属性，且有对应的 get 和 set 方法



## 其它

**package**

* 声明源文件所在包，每 `.` 一次，就是一级文件目录
* `package com.keith`: 说明此源文件在 `com/keith` 目录下



**import**

* 显示导入指定的包下的类或接口

