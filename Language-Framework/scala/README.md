# scala 程序结构

```scala
// object ： 一个伴生对象，就可以理解成为一个对象
// HelloScala 是一个对象的名字，底层真正对应的类名是HelloScala$
// 对象是 HelloScala$ 类型的一个静态对象 MODULE$
// 当编写一个 object HelloScala的时候，底层会生成两个.class文件，分别是HelloScala, HelloScala$， 即：会生成两个class 文件。
// object HelloScala 对应的是 HelloScala$ 的一个静态对象（所以就是单例咯）
object HelloScala{
    // def 表示一个方法，是一个关键字
    // main 表示方法的名字，表示程序的入口
    // args: Array[String] 参数名字在前，类型在后
    // Unit =  表示该函数的返回值为空
    def main(args: Array[String]):Unit = {
        
    }
}
```

## 字符串输出的三种方式

```scala
println("a" + b + "c")
printf("%s, %d", "name", 15)
// 类似 shell的 string语法
println(s"$name, $age")
// 大括号表示， 里面是一个表达式
println(s"${age + 10}")
```



# 变量 & 类型

* scala 中：小数默认为 double，整数默认为 int
* `val` ： 数值不可变，不可修改
* `var`:  变量可以修改
* scala中数据都是对象，没有原生类型
* scala中数据类型分为两大类，AnyVal（值类型）,AnyRef(引用类型)，这两种也都是对象
* AnyVal: 
  * Byte, Short, Int, Long, FLoat, Double, Boolean, Char, StringOps, Uint
* AnyRef:
  * Scala Collections, All Java Classes, Other Scala Classes
* 底层类
  * Null：所有AnyRef的子类，只有一个值 null
  * Nothing: 所有类的子类：在开发中，Noting的值可以赋值给任意类型的对象，常用于抛出异常
  * 这个需要和object理解：object是所有类型的父类
    * java中的object对应scala中的 Any
* 隐式转换：
  * **低精度** 向 **高精度** 自动转换，高到低是需要手动转换的哦。
* 1.2 (Double), 1.2f(Float), 1(Int), 1l(Long)
* 科学记数法: 1e-2, 1E-2

```scala
var a : Int = 100
val b : Int = 100

// 类型推导
var num = 10 //推导为int类型，类型确定之后就不能修改了
num.isInstanceOf[INt] //判断类型
```

```scala
object HelloWorld {

  def main(args: Array[String]): Unit = {
    val dog = new Dog
    dog.age = 18
    dog.name = "hello"
    // 该句报错，因为 dog 是val，相当于是个C++ 的底层 const
    dog = new Dog
  }
}

class Dog {
  // _ 在这里表示 默认值
  var age: Int = _
  var name : String = _
}
```
* Char: 2字节 unicode码
* Boolean: 只有 true 和 false, 不能用 1 或 0
* Unit: 只有一个实例对象 ()
* Null: 只有一个实例对象 null, 任何 AnyRef 的子类
* Nothing: 主要只用来抛异常的
* Any: 相当于 java 的 object
* Char 与 Byte, Short 不能互相隐式转换, 三者可以混合计算, 计算的时候会转成 int 然后计算. Byte 和 Short 可以互相转换

## 强制类型转换
* a.toInt, a.toInt(), 可能会导致精度降低或者溢出
* string 类型转换: 
```scala
val a: Int = 10
val aStr: String = 10 + ""; // +个空串 就 ok 了
aStr.to??
```

# 标识符 (命名规范)

```scala
//  两个操作符 可以作为 命名的开头, 必须是两个.... 为啥要这么搞? 有啥必要吗?
val ++ = "hello world"

// 返引号 可以使得 关键字变成变量使用
val `true` = "hello world"
println(`true`)
```

* `_` 的 N 中功能
```scala
import scala.io._ //引入该包下的所有东西
```

# 运算符
* scala 支持代码块返回值

```scala
val res = {
    90
}
```
* 三元运算符 `val num if(5>4) 5 else 4`
* 命令行输入`val val = StdIn.readLine()`

# 分支控制
* 没有 switch
* for 循环
```scala
if (expression) {

} else if (expression2) {

}

// for 表达式, for 推导式
// 1 to 3 :  两边都是 闭合
// 1 until 3: 前闭 后 开
// 也可以这样直接对集合进行遍历

for(i <- 1 to 3){

}

// 循环守卫, 如果为 false 就直接跳过, 相当于 continue, scala 中没有 continue 和 break
for (i <- 1 to 3 if i!=2) {

}

// for 引入变量
for (i <- 1 to 3; j = 4-i) {

}

// 等价于
for (i <- 1 to 3; ) {
    j = 4-i
    ...
}

// 嵌套循环, 等价于嵌套循环的
for (i<-1 to 3; j <-1 to 3){
}

// 循环返回值, 将遍历过程中产生的值放到一个 vector 中去， 使用yield关键字
val res = for(i<-1 to 10) yield i

// yield + 代码块返回值
val res2 = for(i<-1 to 10) yield {
    if (i%2 == 0)
    	i
    else
    	"not even"
}
```
* while 循环

```scala
while (expression) {
    
}
```

* 循环中断

```scala
// breakable 高阶函数，接收函数的函数
// breakable(op: =>Unit) 接收一个没有形参，没有输出的函数，
// breakable 对 op 抛出的异常 进行捕获处理
// 当传入的是代码块的时候， 一般会将 小括号 转成 大括号
breakable(
    while (i < 20) {
        n += 1
        if (n == 18) {
            break() //没有 break 关键字，只有个函数，该函数只是扔出一个异常
        }
    }
)
// 当传入的是代码块的时候， 一般会将 小括号 转成 大括号
breakable{
    while (i < 20) {
        n += 1
        if (n == 18) {
            break() //没有 break 关键字，只有个函数，该函数只是扔出一个异常
        }
    }
}
```

* 如何实现 continue
  * 使用 if else
  * 也可以使用循环守卫

* 模式匹配: match-case



# 函数式编程基础

* 推荐递归解决问题
* 方法和函数几乎可以等同：定义，使用，运行机制一样
  * 函数的使用方式更加的灵活多样，（方法转函数）
  * scala 中，函数是一等公民，函数也是对象，函数的创建不依赖类或者对象。java中的函数创建要依赖类，抽象类，或者接口。

```scala
object HelloWorld{
    def main(args: Array[String]): Uint = {
        val dog = new Dog
        dog.sum(1, 2)
        val f1 = dog.sum _ // 方法转函数
        f1(1, 2)
        
        // 直接定义函数
        val f2 = (n1: Int, n2: Int) => {
            n1 + n2
        }
        f2(1, 2)
    }
}
class Dog {
    // 此处是方法， 这里还有默认值的 demo
    def sum(n1: Int, n2: Int = 10): Int = {
        n1 + n2
    }
}
```

* 函数的定义
  * `def func_name(paraname: type):ReturnType = {func_body}`
  * `def func_name(paraname: type) = {func_body}`: 表示自动推断返回值类型
  * `def func_name(paraname: type){func_body}`: 表示没有返回值
  * 如果没有return，默认认为执行的最后一句的值作为返回值
  * 如果没有返回值，return不生效

* 函数的调用机制：依旧是压栈出栈来完的
* 细节总结：
  * 函数如果没有形参，调用的时候可以不加括号
  * 省略 `: Type` 会进行返回参数类型推断
    * 使用返回值类型推断的时候 不要使用 `return` 进行返回
  * 如果省略了 `: Type =` ： 表示该函数没有返回值，这时候即使写了 `return` 也是无效的。
    * 如果使用`: Unit = ` ： 也是表示没有返回值，这时候即使写了 `return` 也是无效的。
  * 如果不确定返回值类型，最好使用自动类型推断
  * 可以方法内定义方法，函数内定义函数？
    * 方法内定义的方法 的 **地位** 其实和普通的类方法一个级别的。
  * scala 函数 形参默认是 **`val`** 的
  * 递归程序不能使用类型推断！！！必须指定返回的数据类型
  * 支持可变参数： `def sum(args: Int*)` 
  * **过程**： 没有返回值的函数（方法）

## 惰性函数

* 尽可能延迟表达式求值。惰性集合在需要时提供其元素，无需预先计算他们。
* 优点：可以将耗时的计算推迟到绝对需要的时候。
* scala对惰性计算提供了原生的支持。
* 当函数的返回值声明为 lazy 时，就成了惰性函数，在真正取值的时候才会被调用
* 细节：
  * lazy 只能修饰 val
  * 

```scala
lazy val i = 100 // 这个变量值也是在使用的时候才会真正的分配
lazy val res = func() //这时候并没有实际调用
println(res) // 这时候才会真正调用！
```



# 异常处理

* `throw new Exception`
* `throw` 抛出异常，抛出的是一个 `Nothing` 类型, 可以看做 return 语句了。

```scala
try{
    
}catch{
    // => 关键字： 表示后面是 异常处理代码。
    case ex: ValueError => {}
    case ex: Exception => {}
}finally{
    
} 

def test(): Nothing = {
    throw new Exception("exception occured")
}

```

* throw comment

```scala
@throws(classOf[NumberFormatException])
def f11(a: String){
    a.toInt
}
```



# 面向对象基础

```scala
object Demo{
    def main(args: Array[String]){
        val cat = new Cat
        // 这里实际调用的是 name_$eq 方法来设置值的。
        cat.name = "hello"
        println(cat.name) //这里实际上调用的是 cat.name()
    }
}

class Cat {
    // 默认是 private
    // 同时会生成两个 public 方法 name()负责getter, name_$eq()，负责 setter
    var name: String = "name" //一定是要给初始值的
    var age: Int = _ // _ 表示默认值
}
```

* 基本语法 `[修饰符] class 类名`，修饰符默认 public
* 属性定义：`[修饰符] val name: Type = DefaultValue`
  * 必须显示给初始值
* scala 一个文件可以包含多个类， 默认都是 public 的



## 方法

* 方法和函数一致，在 class 里面是方法，在 object 里面是函数？

## 构造器

* scala 构造器包括 主构造器 和 辅助构造器

```scala
class Name[形参列表] { // 主构造器
    def this(){      // 辅助构造器
        
    }
    def this(){      // 辅助构造器
        
    }
}


// 这样 主构造器就私有化了
class Person private (inName: String, inAge: Int) {
    // 这部分底层实际上是包装到 一个构造函数里的
    var name: String = inName
    var age: Int = inAge
    
    age += 10
    // 这样辅助构造器就私有化了
    private def this(name: String){
        this(name, 10)// 第一行一定要调用主构造器！！！！
        
    }
    
    override def toString: String = {
        
    }
    // 这个地方同样被搞进了 一个构造函数里
    name += "aa"
}
```

* 细节：
  * 主构造器：**实际上是将 除 函数的语句 都包装到一个 构造器里。**
  * 辅助构造器：第一行一定要调用主构造器（直接或者间接）

