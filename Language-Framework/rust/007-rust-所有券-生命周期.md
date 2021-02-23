# rust所有权

rust 所有权规则，**默认情况下**： （保证所有的资源能够正确的释放）
> 所有权规则实际是针对堆内存的，栈内存只是附带影响了而已。

* Each value in Rust has a variable that’s called its owner.
* There can only be one owner at a time. 
* When the owner goes out of scope, the value will be dropped.

来看一下rust是如何保证这些规则的：
* Each value in Rust has a variable that’s called its owner.
* There can only be one owner at a time. 
  * 通过 `=` 运算符的移动语意来 保证。移动的实际操作是：复制栈数据，将原始的owner置为不可用！这样就防止了内存问题 double free
  * 如果 `=` 运算符对于所有的类型，都是移动语意的话，那么对于栈数据就不公平了，我又不会存在 double free 问题，为啥我需要将原始 owner置为 不可用呢？所以 rust 提供了Copy trait来解决这个问题，具有 Copy trait 的类型，`=` 是复制语意。（仅栈上数据复制，并不需要将原始owner置为不可用）
  * 🤔️：为啥不一锅端呢？栈上数据也是移动语意有啥问题吗？
```rust
fn main(){
    let b = String::from("hh2");
    let c = b; // 移动语意，后面，b将不再可用。
}
```

* When the owner goes out of scope, the value will be dropped. （确保分配的内存的会正确的回收）
  * 通过 Drop trait 来确保资源能够正确的回收♻️
```rust
fn main(){
    { 
      let b = String::from("hh2");
    } // b离开作用域，其管理的堆内存会被 drop
    
    let mut b = String::from("hh2");
    b = String::from("hh3"); // 这时，原来b owner 的 空间也会被 drop 的吧。
}
```

## 不可变引用 & 可变引用（借用）

来看 引用 & 借用 需要遵守哪些规则

* At any given time, you can have either one mutable reference or any number of immutable references.
  * 可以多个引用存在
  * 只能有一个借用存在，且借用和引用不能同时存在
* References must always be valid.

为什么要遵守这些规则
* At any given time, you can have either one mutable reference or any number of immutable references.
  * 
