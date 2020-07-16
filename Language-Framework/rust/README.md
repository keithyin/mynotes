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

