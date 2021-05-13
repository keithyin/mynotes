> Rust 是C++的最佳实践

> C++编码中的常见的三个内存问题：1）堆内存的回收，2）循环引用，3）野指针。Rust使用所有权机制缓解了（堆内存的回收），使用生命周期标记解决了（野指针），循环引用还没解决。。。


* 在C++编码过程中，需要注意的其中一点就是**堆内存**的管理，如果不使用RAII方式组织代码的话，很容易就忘记释放**堆内存**了，导致内存泄漏。
* Rust 默认就是 RAII

所有权规则：
* Rust 中的每一个值都有一个对应的变量作为它的 所有者。（一个值只有一个所有者）
* 在同一时间内， 值有且仅有一个所有者
* 当所有者离开自己的作用域时，它持有的值就会被释放掉。PS：rust的作用域与C++一致，`{}` 用来划分作用域

Rust类型系统：
* 值类型: 
* 引用类型
> 和C++对比来说，C++似乎只有值类型，1）指针实际上是值，只是里面存的是地址。2）引用是可以被引用的值一样使用的。而在Rust中，`引用类型`就是`引用类型`，只能引用类型来进行赋值。

# 移动，浅复制，深复制
C++ 中 `=` 表示三个含义，1）如果右边是右值，则是移动含义。2）右边是左值，代码实现是深复制，则是深复制。3）右边是左值，代码实现是浅复制，则是浅复制
Rsut中 `=` 表示一个含义，就是移动（所有权转移）。有两个区别：1）如果右边对象实现了 Copy trait，那么移动之后，那个变量还可以用。2）如果没有实现 Copy trait，移动之后，该变量就废弃了。
Rust如果想深度复制怎么办呢？那就先 `.clone()`，然后再 `=` 转移所有权到新变量。

# mut
mut不能理解为类型标识符。在 `struct` 中声明一个 `mut` 属性，从编译器中的报错信息就可以看出来，那么如何理解 mut 呢？
```rust
struct Demo {
age: mut i32, // 瞅瞅编译器的报错信息
}
```
如果将`值`和`变量`分开理解的话，类型系统是用来修饰`值`的，而`mut`是用来修饰变量的。
* `let mut a = 10;` 表示对于其 管辖的值，我有能力修改之。
* `let a = 10;` 表示对于其 管辖的值，我没能力修改之。
* 管辖含义就是 具有所有权。管辖包括，栈空间 和 堆空间。

```rust
let a = 10;  // a管理了一个栈上空间，里面放了值，10. 因为是 let a，a没有权限去修改栈空间上的值。
let b = &a; // b管理了一个栈上空间&a，&a对应一个栈空间放着指针。 因为是 let b，所以 c 不能修改那个空间的指针值。
let mut c = &a; // c管理了一个栈上空间，值是&a。因为是 let mut c. 所以这个空间的指针可以变，比如后面还可以加上 c = &b;
let mut d = 1000; //d管理了一个栈上空间，值是1000。因为是 let mut d.所以这个空间的值可以变，d = 1;
```

# reference，borrow

* 引用的引用的引用
* 借用的借用的借用

# 结构体
* 由于 mut 不是类型表示符。而 结构体内以下两种写法都不行。岂不就无法使用结构体的方法来修改其中引用所指向的值？
  * rust 不支持单独声明结构体内部分字段的可变性，一旦实例可变，所有字段都可变。但是当实例不可变的时候，所有的字段都不可变
  * 当我们需要不可变实例中包含一个可变属性的时候，可以借用`Refcell`达到这个效果。
```rust
struct Demo<'a>{
  age: &'a mut i32, //这个不行的话，如果修改 结构体引用的外部变量的值呢？
  mut name: &'a String, // 不允许修饰部分字段的可变性
}
```

**多态** 
* 将实例传给trait对象，trait对象，实际就是 `Box<dyn SomeTrait>` `Box`也可以是其他指针？



# rust中的高级类型

* `Never type`: `!`  it stands in the place of the return type when a function will never return
  * `!` never has a value

```rust
fn bar() -> ! {
    // --snip--
}
```



`Never Type` 有什么用呢？

* `match arm` 中用.

```rust
let guess = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue, // 因为 guess 只能有一个确定的类型，所以 continue 返回的是 !
        };
```



`fn main()` 函数很特别，其返回值类型并不能随意指定。其有两个可选的返回值类型 `(), Result<(), E>` .

```rust
use std::error::Error;
use std::fs::File;

// Box<dyn Error>: dyn Error 表示 对象实现了 Error trait。Box表示对象放在了堆上。
fn main() -> Result<(), Box<dyn Error>> {
    let f = File::open("hello.txt")?;

    Ok(())
}
```



## zero-sized type (ZST)

https://runrust.miraheze.org/wiki/Zero-sized_type

https://www.hardmo.de/article/2021-03-14-zst-proof-types.md

