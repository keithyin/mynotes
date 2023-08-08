代码块前使用unsafe可以使代码切换到不安全模式，并在被标记的代码块中使用不安全代码。不安全Rust允许执行4种在安全Rust中不被允许的四种操作。

1. 解引用裸指针
2. 调用 unsafe 的函数或者方法
3. 访问或修改可变的静态变量
4. 实现 unsafe trait

注意：unsafe关键字并不会关闭检查器，也不会禁用任何其他Rust安全检查。这意味着，如果在unsafe中使用引用，那么该引用依旧会被检查。


## 解引用裸指针

* 可变裸指针 `*mut T`
* 不可变裸指针 `*const T`

裸指针 与 引用、智能指针的区别在于：

* 允许忽略借用规则。即 可以同时拥有指向同一地址的 可变 和 不可变指针，或者 拥有指向同一地址的 多个 可变指针
* 不能保证自己总是指向了有效的内存地址
* 允许为空
* 没有实现任何自动清理机制

```rust

let mut num = 5;

//创建指针的时候 无需 unsafe。只是在解引用需要加上 unsafe
let r1 = &num as *const i32;
let r2 = &mut num as *mut i32;

unsafe {
  println("r1 is: {}", *r1);
  println("r2 is: {}", *r2);
}
```

## 调用不安全的函数或方法

如果函数或者方法的签名是 `unsafe fn func()`. 那么就只能在 unsafe block中进行调用了。
不安全的函数和方法中的 `unsafe` 关键字意味着我们在调用该函数前，需要手动满足并维护一些先决条件，因为rust无法对这些条件进行验证。
通过在 `unsafe block` 中调用该函数，我们向Rust表明自己确实理解并实现了相关的约定

```rust
unsafe fn dangurous() {}

fn some_func() {
  unsafe {
    dangurous();
  }

}
```

### 创建不安全的代码抽象

有些操作使用安全的rust是无法完成的，所以我们有时会借助unsafe rust来实现相关功能，然后将其封装在一个安全的函数中。
这隐含着另一种意思，函数中包含不安全代码并不意味着我们需要将整个函数都标记为不安全。我们只需要将不安全代码放在 `unsafe block` 中就可以了


```rust

fn safe_func() {

  unsafe {
    // some unsafe code. dereference raw pointer, .. etc
  }
}

```

### 使用 extern 调用外部代码

某些场景下，rust需要和另外一种语言编写的代码进行交互。
Rust 提供了 `extern` 关键字来 简化 创建 和 使用外部函数接口的过程。

任何在 `extern` 块生命的函数都是不安全的。因为其它语言不会强制执行Rust遵守的规则

```rust
extern "C" {
  fn abs(input: i32) -> i32;
}

fn main() {


}


```

### 其它语言调用rust函数

```rust
// no_mangle 告诉rust，在编译的时候不要改函数的名称。 而且这个block 中不需要用 unsafe
#[no_mangle]
pub extern "C" fn call_from_c() {

}
```

## 访问或修改一个可变静态变量

rust中，全局变量也称之为静态变量

```rust

static HELLO_WORLD: &str = "hello world";

static mut COUNTER: i32 = 0;

fn main() {
  println!("{}", HELLO_WORLD);

  unsafe {
    COUNTER += 1; // 这里的unsafe主要是 担心多线程的 data race问题。 在拥有全局访问的可变数据时，我们很难保证没有 data race事情发生，这也是rust认为 操作可变静态变量是unsafe的原因。
  }

  unsafe {

    println!("{}", COUNTER); // // 这里的unsafe主要是 担心多线程的 data race问题
  }

}

```

## 实现不安全 trait

```rust
unsafe trait Foo {
  // some methods
}

unsafe impl Foo for i32 {

  
}

```
