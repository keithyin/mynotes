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


// 这样 主构造器就私有化了, inName 是局部变量, 如果用 val 修饰一下, otherName 就是个 只读的私有属性了. 如果用 var 修饰, 就变成了可读写属性!
class Person private (inName: String, inAge: Int, val otherName: String, var ootherName: String) {
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
  * BeanProperty: 对于属性 @BeanProperty 就自动生成了其 setXXX 和 getXXX 方法. 原来自动生成的方法也可以使用!
  * 


## 打包 & import
* 子包可以直接使用父包的东西
* 父包使用子包的, 必须要引入
* anywhere can import

```scala
package aa.bb.cc

// 与上面的等价
package aa.bb
package cc

// 一个文件中可以创建多个包, 每个包里面都可以写 class, trait, object
package aa{
    package bb {
        package cc {
        
        }
    }
}
```
* 解决java 中不能在 类外 定义对象的限制: 使用包对象来解决
```scala

// 每一个包都可以有一个包对象
// 包对象的名字需要和子包的名字一致
// 在包对象中可以定义变量和方法
// 包对象中定义的 东西, 可以直接在包中使用了
// 包对象只能在 父包中定义, 和包是同级的!!!
// package object scala{}  创建包对象,里买可以搞事情了.
package object scala {
    
}

// 打包
pakcage scala {
    
}

import scala.collection.mutable._
// 重命名 与 隐藏掉
import scala.collecton.mutable.{HashTable=>JavaHashTable, List=>_, Other}

```

# 伴生类与伴生对象

```scala
// for non-static
// default private
// no public key-word
class Clerk {
    // default private
    var name: String = _
}

// can access private property of the Cleak class
// for static 
object Clerk{

}
```


# scala 集合包

* 同时支持 **可变集合** 和 **不可变集合** ， **不可变集合可以安全的并发访问**
  * 不可变集合：集合本身不可变， 元素个数不可变，但是存储的值可以变换的。
  * 可变集合：可动态增长
* 两个主要的包
  * `scala.collection.immutable`
  * `scala.collection.mutable`
* scala 默认采用不可变集合（几乎所有）。
* 集合有三大类型： `Seq, Set, Map`

## 数组

```scala
// 定长数组, scala中小括号进行索引， [泛型]
val arr = new Array[Int](10)
for (item <- arr) {
    println(item)
}
// 可以修改
arr(3) = 10

// 第二种方式 定义数组，直接初始化数组，数组的类型和初始化的类型的是有关系的
// 这里其实使用的 object Array
var arr02 = Array("1", 1)
for (index <- 0 until arr02.length) {
    arr02(index)
}
```



```scala
// 变长数组
val arr = ArrayBuffer[Int]()
arr.append(7)
arr.append(8)
// 可变参数
arr.append(7,9,0) 
arr(0) = 10
arr.remove(idx)

// 到 定长 和 变长 之间的转换
arr.toArray.toBuffer 
```



## 多维数组

```scala
// 创建
val arr = Array.ofDim[Double](3,4)
arr[1][1] = 10
for (item <- arr) {
    for (item2 <- item) {
        
    }
}

```



## 元组

```scala
// 元祖最多能放 22 个元素
// 类型 是 Tuple4
val tuple1 = (1, 2, 3, 4, "HELLO")

// 访问 元祖的第一个元素, 有两种方式
tuple1._1 
tuple1.productElement(0)

// 遍历, 需要使用到迭代器
for (item <- tuple1.productIterator) {
    
}
```



## List

* `List` 只有 不可变的, 可变的是 `ListBuffer`
* `Nil` 一个空集合

```scala
val list = List(1, 2, 3)
val nullList = Nil // 空集合

// 访问
list(1)

// 遍历
for (item <- list) {
    
}

// 追加元素， 原 list 没有变， 只是返回一个新的 list
// 注意 + 的位置， 靠近 加的值
val list2 = 4 +: list
val list3 = list2 :+ 5

// Nil 一定要放到最右边， 最右边的一定是一个集合。
// 得到的结果是 从左到又 依次放到 Nil 集合中去
val list5 = 1 :: 2 :: 3 :: list3 :: Nil
// ::: list 内容展开然后放到 集合中去
val list6 = 1 :: 2 :: 3 :: list3 ::: Nil
```

* ListBuffer: 可变的 list 集合

```scala
var lb = ListBuffer[Int](1,2,3)
// 访问
lb(2)

// 遍历
for (item <- lb) {
    
}

// 添加
lb += 10
lb.append(11)
lb ++= lb2 // 一个集合加到另外一个集合中去， 元素级别相加
val lst2 = lst1 ++ lst3 // 同上
val lst5 = lst :+ 5 // lst 不变
val nullLb = new ListBuffer[Int]

// 删除
list.remove(10)
```



## 队列 Queue

* 先入先出

```scala
val q = new mutable.Queue[Int]

// 增加， 默认是增加到 队列的屁股后面的。
q += 1

q ++= list // 将 list 中的元素批量添加
q += list // 将 list 整体加到 queue 中去

// 入队列 与 出队列
q.dequeue()
q.enqueue(1,2,3)

// 队列的 头 和 尾
q.head // 第一个元素
q.last // 最后一个元素
q.tail // 返回除了第一个元素以外的所有元素 (返回的是一个队列)
t1.tail.tail // 
```



## Map

* 存的是 key-value
* 常用的是可变的版本
* 不可变的版本 是有序的。
* map 的底层是  Tuple2 类型

```scala
val map = mutable.Map("a"->1, "b"->2)
val map2 = new mutable.Map[String, Int]
val map3 = mutable.Map(("A", 1), ("B", 2))
// 访问, 
val val1 = map2("a") // 如果不在里面， 会抛异常
map2.contains("a") // 判断key 是否存在
map2.get("a").get // 如果key存在 map2.get("a") 返回的是一个 Some， 这时候 再 get 一次即可， 如果key不存在，map2.get("a") 返回的是 None
map2.getOrElse("a", default_value) // 有则返回， 否则 返回默认值
// 遍历
for ((k, v) <- map2) {
    
}

for (k <- map2.keys) {
    
}

for (v <- map2.values) {
    
}


for (kv <- map2) {
	// kv 是 Tuple2    
}

// 增加, 存在就更新， 不存在就添加， 如果是 immutable,Map， 值都不让改！！！
map2("AA") = 20
map2 += ("D"->4)
map2 += ("D"->4, "E"->5)
// 删除, 直接写 key 就可以了， key不存在 也不会报错
map2 -= ("D", "E")

```



## Set

```scala
val set = Set(1, 2, 3)
val mutableSet = mutable.Set(1, 2, 3)

// 添加
mutableSet += 4
mutableSet += (4)

// 删除
mutableSet.remove(2)
mutableSet -= 4

// 遍历
```



# 集合操作

* map 操作

```scala
// 高阶函数
def test(f: Double => Double, ni: Double) {
    f(n1)
}

// 无参的高阶函数
def teset2(f: ()=>Unit) {
    
}

def myPrint() {
    println("hello world")
}

// 为什么要有 _ 这个神奇的操作， 因为 scala 中对于 无参的函数，可以不加 () 直接调用，所以如果想将一个 函数赋值一个变量的话，那就需要 后面显式的加上一个 _ 高速编译器，不要计算函数的值。
val f1 = myPrint _
```

* `flatMap`: 如果遍历的元素是个集合， 那就继续展开

* `reudceLeft, reduceRight` 从左开始算， 从右开始计算
*  `foldLeft, foldRight`: 

```scala
val list3 = List(1, 2, 3, 4)
// 等价于 list 3 左边 加上一个元素 5， 然后执行 reduce
val l4 = list3.foldLeft(5)(_ - _)
// 等价于 list3 右边 加上衣蛾元素 5， 然后执行 reduce
val l5 = list3.foldRight(5)(_ - _)

val l7 = (1 /: list3)(_ - _) // 等价于 list3.foldLeft(1)(_ - _)
val l8 = (list3 :\ 1)(_ - _) // 等价于 list3.foldRight(1)(_ - _)

```

* `scanLeft, scanRight` : 保存中间结果 的 `fold...`

```scala

```

* `zip`
  * 两个list个数不一致，则会导致数据丢失

```scala
val list1 = List(1, 2, 3)
val list2 = List(2, 3, 4)
val list3 = list1.zip(list2) // [(1, 2), (2,3)] 出来的是 Tuple2
```

* 迭代器, `list.iterator()`
  * `hasNext(), next()` 
  * 迭代器可以直接放到 for loop 中去

* `Stream`
* `.view` 方法 : 懒加载机制！！

```scala
// 这时候时候并没有执行 filter
val viewdemo = (1 until 10).view.filter(a => a%2 == 0)
println(viewdemo) // 这时候也没有调用
for (item <- videwdemo) {
    // 遍历的时候才会真正的调用。
}
```

# 操作符重载
```scala
class Dog {
    def -(i: Int) {
    
    }
    // dog++, 后置操作符重载
    def ++() {
        
    }
    // !dog   unary 一定要加上, 表示是 前置运算符
    def unary_!() {
    
    }
}

// 中置操作符: a op b  等价于 a.op(b)
```

# 模式匹配
* match case 模式匹配, 两个都是关键字
* 如果都没有匹配上, 就会匹配 `case _`
* 如果没有匹配上, 且没有 `case _` 那就回抛出异常
* 

```scala

variable match {
    case '+' => res = n1 + n2 // 默认是 break 的
    case '-' => {
        res = n1 - n2
    }
    case _ => res = n1
}

```
* match 中的守卫
```scala
variable match {
    case _ if (expression) => do something
    case _ => do something
    
}

varable match {
    // 意味着 mychar varable, 这个一直是会匹配上的
    case mychar => do something
}

// 可以有返回值哦
val res = vari match {

}

```
* 类型匹配

```scala
obj match {
    // obj 给 a, 然后看 a 是不是 Int ? 
    case a: Int => a
    case b: Map[String, Int] => b
    // 如果 _ 出现的不是最后一个, 表示的隐藏变量名, 而不是默认匹配
    case _: Double => "double"
    case _ => "nothing"
}
```

* 数组匹配

```scala

arr match {
    // 匹配只有一个元素, 且第一个元素为 0 的数组
    case Array(0) => '0'
    // 匹配有两个元素的数组, 并将两个元素 赋值给 x, y
    case Array(x,y) => x + "=" + y
    // 匹配 以 0 开头的数组
    case Array(0, _*) =>
    
    case _ => ""
}

```

* 列表匹配

```scala
list match {
    // 只有 0 的
    case 0 :: Nil => '0'
    
    case x::y::Nil => x + " " + y
    // 0 打头的
    case 0::tail => "0..."
    case _ => "nothing"

}
```

* 匹配元祖
```scala
tuple match {
    // 第一个元素是 0 的二元组
    case (0, _) => 
    // 第二个元素是 0 的二元组
    case (y, 0) =>
}

```

* 对象匹配

