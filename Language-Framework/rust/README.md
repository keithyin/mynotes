# 命名风格

* 类名：驼峰命名
* 变量 & 函数：全小写 + 下划线
* 常量：全大写 + 下划线

# cargo 使用

* 创建工程: `cargo new projname`
* 编译代码: `cargo build`
* 语法检查: `cargo check`
* 运行 `cargo run` (编译之后才能操作？)

```rust
fn main() {
    // 打印函数的 宏
    println!("hello world!");
}
```



# 变量

```rust
fn main() {
    // 变量定义 let name: type = init_value; 省略 type 的话 类型可以自动推导。
    let a = 1;
    let b: u32 = 2;
    println!("a = {}", a)
    b = 2; // 会报错。
    let mut c: u32 = 3;
    c = 3; // 手动指定 mutable，就可以赋值了。不指定mut的 都是不可以修改的。
    
    // 变量的隐藏性, 同样的代码段里，可以对变量名重定义。
    let b: f32 = 1.1;
    
    // 常量
    const MAX_POINT: u32 = 10000;
    
    // 常量 vs immutable variable ？
}
```

类比 顶层 const 与 底层 const 来理解变量声明与定义，实际上只有指针才区分 顶层 const 还是 底层 const。对于值，就只有顶层 const. 因为 rust 称之为 immutable, 所以就使用 顶层 immutable 与 底层 immutable 来说明吧

```rust
let a = 10; // a 顶层 immutable
let ref_a = &a; // 顶层 immutable + 底层 immutable
let mut mut_ref_a = &a; // 顶层 mutable + 底层 immutable

let mut b = 100; // 顶层 mutable
let ref_b = &b; // 顶层 immutable + 底层 immutable
let mut mut_ref_b = &b; // 顶层 mutable + 底层 immutable
let ref_b = &mut b; // 顶层 immutable + 底层 mutable
let mut mm_ref_b = &mut b; // 顶层 mutable + 底层 mutable

// 引用的引用
let mut a = 1000;
let mut ref_a = &a;
let ref_ref_a = &ref_a;
println!("{}", **ref_ref_a);
```

所以对于 等号（=） 做一个简单的总结
* 等号 右边
    * 如果右边是值：执行移动语义 或者 赋值语义
    * 如果右边是引用：&是赋值语义，&mut 是移动语义。只是语义在，我发现并不会做严格检查
* 等号 左边，等号左边就简单了，要么就是 let，要么就是 let mut
    * let: 顶层 immutable
    * let mut: 顶层 mutable
* 右边确定了 变量的部分类型，左边只是考虑是不是给变量 加上 顶层 immutable

顶层 immutable
```rust
let a = 10;
a = 100; // fail, 顶层 const

let a = 10;
```

```rust
{
        let mut a = 1000;
        let mut ref_a = &mut a;
        let ref_ref_a = &mut ref_a; // 这里去掉 mut，就会报错，思考一下为什么。
        println!("{}", **ref_ref_a);
        **ref_ref_a = 100000;
        println!("{}", **ref_ref_a);

    }
```

* 思考以下代码。函数参数列表中的 变量 一定是 let 的。而非 let mut 的。
```rust
fn modify(a: &mut &mut i32) {
    **a = 100;
}

fn call_modify(){
    let mut a = 1000;
    let mut ref_a = &mut a;
    let ref_ref_a = &mut ref_a;
    modify(ref_ref_a);
    println!("modified: {}", a);
}
```

# 基础数据类型

```rust
fn main() {
    let is_true: bool = true;
    let is_false = false;
    println!("{}{}", is_true, is_false);
    
    //char, rust里， char 是32位的。所以 char 表示的是 unicode
    let a = '你';
    println!("{}", a)
    
    // 数字类型： i8, i16, i32, i64, u8, u16, u32, u64, f32, f64
    // 自适应类型（平台相关的类型，长度与平台有关）：isize, usize ??
    
    // 数组 [type; size]，size 也是数组类型的一部分，注意后面数组是 [] 包起来的哦
    let intarr: [u32, 5] = [1, 2, 3, 4, 5];
    
    // 元祖
    let tup: (i32, u32, char) = (-3, 3.3, '天');
    println!("{}", tup.0); // 打印元祖的第一个元素
    let (x, y, z) = tup; // 元祖的 tie 操作。
    
}

// c/c++中 函数传参 给数组的时候 会退化成指针，所以 数组的大小不起作用，但是在 rust中。
// 数组的长度也是类型的一部分，一个 [u32; 5] 的数组 传给一个 [u32; 3] 的形参，会报错
fn print_array(arr: [u32; 3]) {
    for i in &arr {
        println!("{}", i)
    }
}
```



## 字符串

```rust
let s = String::from("hello");
```





# 函数



```rust
// 形参一定是要带着 类型的。
fn func1(a: i32, b: u32) {
    
}

// 尾部返回值类型
fn func2(a: i32, b: i32) -> i32 {
    return a + b;
}

// 
fn func2(a: i32, b: i32) -> i32 {
    // 最后一个语句的 值就是返回值, 这里不能用 分号。。。
   a + b
}


fn func3() {
    let y = {
        let x  = 1; // 语句，只执行操作，不返回值
        x + 1   // 表达式，返回值 （注意后面不加分号。）
    };
}
```



# 控制流

```rust
fn main() {
    let y = 1;
    
    if y == 1 {
        
    } else if y == 0 {
        
    } else {
        
    }
    // 这种写法，两个分支必须是同一个类型。
    let x = if condition {
        5
    } else {
        6
    };
    
    let mut counter = 0
    loop {
        if counter == 10 {
            break;
        }
        counter += 1;
    }
    
    // break 的时候可以返回值
    let result = loop {
      counter += 1;
      if counter == 20 {
          break counter * 2;
      }
    };
    
    while counter < 100 {
        counter += 2;
    }
    let arr: [u32; 4] = [1, 2, 3, 4];
    for ele in &arr{
        
    }
    
}
```



# 所有权

* rust 通过 **所有权机制 管理内存**，编译器在 **编译的时候** 就会根据 **所有权规则** 对内存的使用进行检查
* **所有权是谁的所有权** ：这里我的理解是　**堆 memory 的 所有权**，因为 栈memory的回收是不需要程序员考虑的，出栈了必定会被回收。
* 作用域： {} 是一个作用域，这个和 C/C++ 一样。
* 所有权规则
  * Each value in Rust has a variable that’s called its *owner*.
    * 有变量名的值 才有 owner，值的owner为变量名
  * There can only be one owner at a time.
    * 一个值在同一时刻 只能有一个 owner
  * When the owner goes out of scope, the value will be dropped.
    * owner 出了作用域之后，对应的 值 会被销毁
  * 这里 值 表示的是 堆数据？？

```rust
fn main() {
    {
        let x: u32 = 10;
    }
    println!(x); // 错误 x 定长， x 出了作用域就会被销毁
}
```



* String内存回收： 离开的时候会调用 drop

```rust
fn main() {
    {
        // 存储也是堆上放实际数据， 栈上存放 长度 和 指针。
        let s = String::from("hello");
        // String 类型离开作用域的时候会调用 drop方法
    }
} 

```

## 移动： （move）

* 这个移动其实和 c++ 中的概念是差不多的
* 如何判断 移动后的变量是否可以使用？
  * 如果 变量 是 堆内存的 owner，那么移动之后就不可再使用
  * 如果变量 仅仅是 栈内存的 owner，那么移动后可以继续使用。所以这个移动操作 对于 栈上的数据来说，仅仅是 copy

```rust
fn main() {
    let s1 = String::from("hello");
    
    // 首先：这里是个浅拷贝，再者，string离开作用的域的时候会调用drop方法进行内存回收
    // 如果没有特殊处理的话，会导致 s1 s2 共同指向的空间释放两次。
    // 所以rust 对这个操作进行了重新处理：let s2 = s1 表示的含义就变成了 c++ 中的 move 操作。
    // 所以这个移动是 只有对堆有效吗？
    let s2 = s1;
    println!(s2);
    println!(s1); // 这里会报错： borrowed after move
    
    let x = 5;
    let y = x;

    println!("x = {}, y = {}", x, y); // 这里就可以直接运行
}
```

* 栈上数据拷贝
  * 具有 copy trait 的类型，在 `=` 之后都可以继续使用。
  * 常用 具有 copy trait 的类型有：
    * 整数，浮点型，bool，字符类型，元祖

```rust
fn main() {
    // 栈上的数据，拷贝是安全的。
    let a = 1;
    let b = a;
    println!("{}{}", a, b);
    // 只要 类型实现了 copy trait， 那么拷贝之后就是可以使用的。默认的情况下 堆上的东西，拷贝了就不能用了。
}
```

* 函数和作用域

```rust
fn take_ownership(some: String) {
    
}

fn makes_copy(i: i32) {
    
}

fn take_and_return_ownership(some: String) -> String {
    some
}


fn main() {
    let s = String::from("hello");
    take_ownership(s);
    // 这个位置 s 已经不能用了，因为调用了函数，s 已经给 函数了
    let x = 5;
    makes_copy(x);
    let s2 = String::from("hello");
    let s3 = take_and_return_ownership(s2);
    //s3 在这里就是可以用的咯
}
```

* 所有权 与 函数

```rust
fn gives_ownership() -> String {
    // 返回的时候，所有权转移
    let s = String::from("owner");
    s
}

fn takes_and_gives_back(s: String) -> String {
    // 所有权给到 s， 然后 s的所有权再转移出去。这里的所有权是谁的所有权？
    s
}

fn main() {
	let s = gives_ownership();
    
}
```

## 引用

* 语义： 
  * 原始对象的引用，而非所有权转移
  * 不能通过引用来修改原始对象：因为不具有所有权，所以不允许修改
  * 形参与实参：形参 `param: &Type` , 实参传递的时候也必须是引用，对于非引用 `obj` 使用 `&obj`, 对于引用 `obj` 直接传就好了。如果将引用也看做一个类型的话，上述的规则就比较容易理解了。这点和 `c++` 不一样。
* 语法
  * `&obj` ： 获取对象的引用 （其实也就是指针。）
  * `*(&obj)`: 解引用。目前不知道是干嘛用的。

```rust
fn len(s: &String) -> usize {
    s.len()
}

fn main() {
    let s = String::from("hello");
    let num_ele = len(s); // 错误，这里不能这么传递！！！
    
    // &s; 创建一个指向值的引用，但是并不拥有它，因为不拥有这个值，所以当引用离开作用域时不会释放原始值！！！
    let num_ele = len(&s); // 正确： 这里是正确的传引用的方式（也是取一个地址用？）。。。。。 和 c++ 的说法区分一下
}
```



```rust
fn modify(s: &String) {
    s.push_str(", world"); // 这里会出错， 因为
}

fn main() {
    let mut s = String::from("hello");
    // 这里会报错， 因为语义上，引用不拥有那个值，所以引用也不能修改那个值！！！
    modify(&s);
}
```

## 借用 borrow / mutable reference

* 语法：见代码片段
* 语义：
  * 在一个特定的scope中，一个对象 最多只能有一个 borrow
  * 有了 mutable reference 之后，之前的 immutable reference 和 原始 obj 都不能被使用

```rust
fn modify(s: &mut String) {
    s.push_str(", world");
}

fn main() {
    let mut s = String::from("hello");
	modify(&mut s);
    // 一旦被借用之后，就不能再用原来的 变量了。 之前的引用也不能再使用了。
    // 借用之后，也不创建新的引用了。
}
```

## Slice （a different kind of reference）

* 语法：见代码片段
* 语义：
  * `slice` 不具有 ownership
  *  Slices let you reference a contiguous sequence of elements in a collection rather than the whole collection.
  * 

```rust
fn main () {
    let s = String::from("hello world");
    let h = &s[0..5]; // 左闭右开
    let h = &s[0..=4]; // 左闭右闭
    let h = &s[..5]; // 左闭右开
    let h = &s[..=4]; // 左闭右闭
    let h = &h[6..]; // 到末尾
    let h = &h[..]; // 从头到尾
    
    let h = "hhhh"; // 字面值就是一个 slice，不可变的引用。 结构体（len, begin, cap）
    let h = [1, 2, 3, 4];  // 定义的一个数组， 也是一个 slice
    
    
}
```



# 结构体

```rust

struct User {
	name: String,
    count: String,
    nonce: u64,
    active: bool,
}

fn main() {
    let xiaoming = User {
      	name: String::from("xiaoming"),
        count: String::from("90000"),
        nonce: 1000,
        active: true,
    };
    
    let mut xiaohuang = User {
      	name: String::from("xiaohuang"),
        count: String::from("90000"),
        nonce: 1000,
        active: true,
    };
    
    xiaohuang.nonce = 20000;
    
    // 创建结构体时：如果参数名字 与 字段名字 重名的话， 可以将 字段名字省略掉
    
    // 从一个结构体创建另外一个结构体
    let user2 = User{..user1}; 
    
    // 元祖结构体：（1）字段没有名字，只有类型 （2）使用的圆括号而非花括号
    struct Point(i32, i32);
    let a = Point(1, 2);
    let b = Point(2, 3);
    println!("{}", a.0);
    
    
    // 没有任何字段的结构体
    struct A{};
}

```

* 结构体打印

```rust
#[derive(Debug)]
struct User {
	name: String,
    count: String,
    nonce: u64,
    active: bool,
}

fn main() {
    let xiaoming = User {
      	name: String::from("xiaoming"),
        count: String::from("90000"),
        nonce: 1000,
        active: true,
    };
    println!("{:?}", xiaoming);
    println!("{:#?}", xiaoming); // 这里打印会自动换行
}
```



# 方法

```rust
struct Dog {
    name: String,
    weight: f32,
};

//  方法
impl Dog {
    fn get_name(&self) -> &str{
        &(self.name[..])
    }
    fn get_weight(&self) -> f32 {
        self.weight
    }
}

// 可以用多个 impl
impl Dog {
    //...
}

```



# 枚举类型与模式匹配

```rust
enum IpAddrKind {
	V4,
	V6,
} // 在外面的时候不需要分号哦

struct IpAddr {
    kind: IpAddrKind,
    address: String,
}

enum IpAddrKindStr {
    V4(String),
    V6(String),
}

enum Message {
    Quit,
    Move{x: i32, y: i32},
    Write(String),
    Change(i32, i32, i32),
}

// 枚举的方法, *解引用

impl Message {
    fn print(&self) {
        match *self {
            Message: Quit => println!("quit"),
            // 值可以解到 x，y 中
            Message::Move{x, y} => println("{}, {}", x, y),
            Message::Change(a, b, c) => println!("{}{}{}", a, b, c),
            _ => println!("write")
            
        }
    }
}

fn main() {
    let i1 = IpAddr {
        kind: IpAddrKind::V4,
        address: String::from("127.0.0.1"),
    };
    // 这里感觉有点像 c++ 中的 union
    let i1 = IpAddrKindStr::v4(String::from("hello"));
}
```

## Option

* 标准库中定义的枚举类型

```rust
// enum Option<T> {
//     Some(T),
//     None,
// }
// 主要是用来 空指针判断的？

fn compute(x: Option<i32>) -> Option<i32> {
    
}

fn main() {
    let num = Some(5);
    let absend_number: Option<i32> = None;
    let val = match num {
        Some(i) => {i},
        None => {0},
    };
    
    // 如果有值，走第一个分值，如果是 None 走第二个分支
    if let Some(value) = val {
        println!("fsd");
    } else {
        println!("None returned");
    }
}
```



# Vector

```rust
fn main() {
    let mut v: Vec<i32> = Vec::new();
    v.push(1);
    // 使用宏创建
    let v = vec![1, 2, 3];
    
    // 离开作用域之后依旧是会整体 drop。
    
    // 增 push
    v.push(1);
    v.push(2);
    v.push(3);
    
    // 删 
    
    // 改
    
    // 查
    let one: &i32 = &v[0];
    
    match v.get(1) { // get 传的是下标
        Some(value) => println!("{}", value),
        None => println!("else"),
    };
    
    // 遍历, 分为不可变的遍历 和 可变的遍历
    // 不可变遍历
    for i in &v {
        println!("{}", i);
    }
    
    for i in &mut v {
        *i += 1;
    }
    
}
```

* `Vec` 可以通过枚举放不同的类型的值。



# String

```rust
// 1： 创建一个空String
// 2: 通过字面值创建一个String：使用 String::from()，使用 str 的方式
// 3. 更新 String， push_str, push, + , format
// 4. String 的索引
// 5. str 索引
// 6. 遍历：chars，bytes

fn main() {
    // 创建一个空字符串
    let mut s0 = String::new();
    s0.push_str("hello");
    
    // 创建一个有初始值的
    let s1 = String::from("hello");
    let s1 = "hello".to_string();
    
    // 更新字符串
    let mut s2 = String::from("hello");
    s2.push_str(" world");
    s2.push('c'); // 添加一个字符
    let s3 = s0 + &s2; // s0 的所有权会移交给 s3.
    
    let s_fmtted = format!("{}-{}-{}", s0, s1, s2); // format 宏 和 print！ 是一样的， 不会剥夺所有权。
    
    // 遍历
    //chars
    let s4 = String::from("你好");
    for c in s4.chars() {
        println!("{}", c) // 能把 你， 好，打印出来
    }
    for b in s4.bytes() {
        println!("{}", b) //按照字节打印，打印的是每个字节的的值
    }
    
    // String
    
    
}
```



# HashMap

```rust
// 1. HashMap<K, V>
// 2. 创建 hashmap
// 3. 读取 
// 4. 遍历
// 5. 更新
use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, i32> = HashMap::new();
    scores.insert(String::from("bob"), 10);
    scores.insert(String::from("hh"), 11);
    
    let keys = vec!(String::from("hello"), String::from("world"));
    let vals = vec!(10, 20);
    let scores: HashMap<_, _> = keys.iter().zip(vals.iter()).collect();
    
    let v = scores.get(&String::from("hello"));
    
    // 遍历
    for (key, value) in &scores {
        println!("{}-{}", key, value);
    }
    println!("{:?}", scores);
    
    // 没有的时候才会插入
    scores.entry(String::from("bb")).or_insert(4);
    

}
```



# 包

* 包：`cargo` 的给一个功能，允许 构建，测试，分享create
* create：一个模块的树形结构，形成库或二进制项目
* 模块：通过 use 来使用，用来控制作用域和路径的私有性
* 路径：一个命名例如结构体、函数或模块等项的方式。
* 包与create
  * 包提供一系列功能的一个或多个create
  * create root 是 src/main.rs 或者是 src/lib.rs。如果只有 main.rs，则说明这个包只有一个 create。如果同时包含 main.rs 和 其他的 lib.rs，则说明有多个 create
  * create 会将一个作用域的相关功能 分组到一起，使得该功能可以很方便的在多个项目之间共享
* 使用模块控制作用域和私有性
  * 创建一个 lib 可以通过 `cargo new --lib libname` 来进行创建
  * 默认所有项（函数，方法，结构体，枚举，模块，常量） 都是私有的，需要使用 `pub` 才能暴露给外部

```rust
mod factory {
    mod factory2 {
        fn produce() {
            
        }
    }
}

mod public_mod {
    pub mod public_mode_inner {
        pub produce() {
            
        }
    }
}

fn main() {
    factory::factory2::produce();
}
```



