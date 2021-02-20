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

**如何实现Copy**

* 使用 `derive`，不能自己实现 Copy。

```rust
#[derive(Copy, Clone)]
struct MyStruct;
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

# `std::clone::Clone`

* A common trait for the ability to explicitly duplicate an object
* 因为我们不能自定义Copy，但是可以自定义Clone。所以Clone比Copy更加灵活。所以可以使用 Clone 来替代 Copy的作用。
* 所以在rust中，Copy只有一个语意：bit-wise copy。但是 Clone 可以有两个语意：深Clone，浅Clone

**实现Clone**

* 如果结构体的所有的字段都实现了Clone，那么可以使用 `#[derive(Clone)]`
* 手动撸

```rust
#[derive(Clone)]
struct MyStruct;
```

```rust
struct MyStruct{
...
}

impl Clone for MyStruct {
    fn clone(&self) -> MyStruct {
        // 代码随便写，返回的不是 MyStruct 都可以。。。。。
        ...
    }
}
```

# `std::marker::Send`

* 用来做类型标记：Types that can be transferred across thread boundaries. （？？如何理解）
* 不用我们管，编译器决定是否实现该trait。This trait is automatically implemented when the compiler determines it's appropriate.
*  An example of a non-Send type is the reference-counting pointer rc::Rc. If two threads attempt to clone Rcs that point to the same reference-counted value, they might try to update the reference count at the same time, which is undefined behavior because Rc doesn't use atomic operations. Its cousin sync::Arc does use atomic operations (incurring some overhead) and thus is Send。
*  Arc：atomic reference counting

# `std::marker::Sync`

* 用来做类型标记：Types for which it is safe to share references between threads. (线程间共享 reference 是安全的)
* 不用我们管，编译器决定是否实现该trait。This trait is automatically implemented when the compiler determines it's appropriate.
* The precise definition is: a type `T` is Sync if and only if `&T` is Send. In other words, if there is no possibility of undefined behavior (including data races) when passing &T references between threads. 
    * 当传 `&T` 给不同的线程时，不会造成 undefined behavior。那么
* The precise definition is: a type `T` is Sync if and only if `&T` is Send. In other words, if there is no possibility of undefined behavior (including data races) when passing &T references between threads. 当传 `&T` 给不同的线程时，不会造成 undefined behavior就
* The precise definition is: a type `T` is Sync if and only if `&T` is Send. In other words, if there is no possibility of undefined behavior (including data races) when passing &T references between threads. 当传 `&T` 给不同的线程时，不会造成 undefined behavior

