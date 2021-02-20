# `use std::ops::Deref`

> 引用相关？

* Used for immutable dereferencing operations, like *v.
* Deref coercion.

```rust
use std::ops::Deref;

struct DerefExample<T> {
    value: T
}

impl<T> Deref for DerefExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

let x = DerefExample { value: 'a' };
assert_eq!('a', *x);
```

# `std::ops::DerefMut`

> 借用相关？

* Used for mutable dereferencing operations, like in *v = 1;
* Deref coercion

```rust
use std::ops::{Deref, DerefMut};

struct DerefMutExample<T> {
    value: T
}

impl<T> Deref for DerefMutExample<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for DerefMutExample<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

let mut x = DerefMutExample { value: 'a' };
*x = 'b';
assert_eq!('b', *x);
```

# `std::marker::Copy`

* 将一个类型标记为 Copy。说明 Types whose values can be duplicated simply by copying bits.（仅拷贝栈上数据即可！）
* A type can implement Copy if all of its components implement Copy
* 结构体只有标记了 Copy，`=` 才是 Copy语意，否则还是移动语意。

* Clone 是 Copy 的超trait，所以，实现了 Copy 的 类，必须同时要实现 Clone，所以 `#[derive(Copy, Clone)]`, 中`Copy, Clone`总是一起出现。
* 如果一个 类型实现了 Copy trait，那么其 Clone trait 的实现仅返回 `*self` 即可

**实现copy的两种方法**

* 使用 `derive`
* 自己撸

```rust
#[derive(Copy, Clone)]
struct MyStruct;
```

```rust
// Copy 一般是由 unsafe 代码实现。
impl Copy for MyStruct {...}
```

```rust
#[derive(Copy, Clone)]
struct Demo{
    // a_str: String, //注释打开后，如果标记 Copy，会报错
    a: i32,
    b: i32
}
fn main(){
    let demo = Demo{a: 14, b: 19};
    let demo2 = demo;
    println!("{}", demo.a); // 如果不标记 Copy，这里会报错，报错信息为 demo 已经被 borrow 了。
}

```
