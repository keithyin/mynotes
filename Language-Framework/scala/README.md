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

```
* 模式匹配: match-case


# 方法

* scala 中如果一个方法没有形参，那就可以省略其调用时的`()`
* 



# 异常处理

* `throw new Exception`

