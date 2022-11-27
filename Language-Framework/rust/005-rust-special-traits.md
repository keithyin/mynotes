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

[rust - Understanding the Send trait - Stack Overflow](https://stackoverflow.com/questions/59428096/understanding-the-send-trait)[rust - Understanding the Send trait - Stack Overflow](https://stackoverflow.com/questions/59428096/understanding-the-send-trait)



* 用来做类型标记：Types that can be transferred across thread boundaries. （？？如何理解）
  * 是否可以在线程间 `安全的传递` 所有权
  * 线程中传递并不是意味着内存位置发生变化，而是所有权发生了变化（即：交给了不同的线程操作这块内存）。
  * Not everything obeys inherited mutability, though. Some types allow you to have multiple aliases of a location in memory while mutating it.
  * Unless these types use synchronization to manage this access, they are absolutely not thread-safe. Rust captures this through the Send and Sync traits.
  * 对于有 multiple aliases of a location in memory 的类型(Rc<T>， &T, ...) 如果他们有操作是 not thread-safe 的，那么就不是 Sync 的。
* 不用我们管，编译器决定是否实现该trait。
* An example of a non-Send type is the reference-counting pointer rc::Rc.  如果两个线程同时尝试 `Clone` 指向同一个值的 Rc, 因为 `Rc` 的引用计数实现不是 `atomic` 的，那么会导致 `data race` 。If two threads attempt to clone Rcs that point to the same reference-counted value, they might try to update the reference count at the same time, which is undefined behavior because Rc doesn't use atomic operations. Its cousin sync::Arc does use atomic operations (incurring some overhead) and thus is Send。
* Arc：atomic reference counting
* `Rc`: `!Send, !Sync`
* 



# `std::marker::Sync`

* 用来做类型标记：Types for which it is safe to share references between threads. (线程间共享 reference 是安全的)
* 是否可以在线程间安全的 `共享变量`
* 不用我们管，编译器决定是否实现该trait。
* The precise definition is: a type `T` is Sync if and only if `&T` is Send. In other words, if there is no possibility of undefined behavior (including data races) when passing &T references between threads. 
  * 解释1: 当传 `&T` 给不同的线程时，不会造成 undefined behavior。那么 T 就是 Sync
  * 解释2: 当 `&T` 是 Send，那么 `T` 就是 Sync



如何简单的判断一个类型是不是 `Send, Sync呢？`



`Sync` 和 `Send` 是一起来保证线程安全的。但是`Sync` 是协助作用。`Send` 才是守门员。因为只要无法`Send` 那么是不会发生多个线程操作同一块区域的情况的！

举例：`Cell, RefCell` 是 `!Sync` 的，因为多个线程操作之是有风险的，所以 `Arc<Cell>` 也会是`!Send` 的，这样就不会出现不同线程操作同一个 `Cell, RefCell` 的问题的。




