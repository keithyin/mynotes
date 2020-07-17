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

* rust 通过所有权机制 管理内存，编译器在 **编译的时候** 就会根据 **所有权规则** 对内存的使用进行检查
* 堆和栈 （与C，C++一样）
  * 如何判断 rust 的数据是分配在栈上 还是 堆上？
    * 定长的变量是放在 栈上的，编译时，长度不固定的变量 是放在堆上的。
* 作用域： {} 是一个作用域，这个和 C/C++ 一样。

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

* 移动

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
}
```

* clone

```rust
fn main() {
    let s1 = String::from("hello");
    // 深拷贝咯
    let s2 = s1.clone();
}
```

* 栈上数据拷贝
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



* 说的所有权，说的是谁的所有权？

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





# 引用

* 使用引用，而非所有权转移

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



* 借用（borrow）: 形参 和 实参 加个 `mut` 就可以了。借用是什么语义呢？

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



# Slice



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
    let h = [1, 2, 3, 4];  // 定义的一个数组
    
    
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
    
}
```

