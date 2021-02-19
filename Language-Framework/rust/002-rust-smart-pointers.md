* Box<T> : point to data on heap。独占所有权
* Rc<T>: Referenced countered smart pointer。共享内部数据所有权
* RefCell<T>: interior mutability pattern。独占内部数据所有权？

# Box<T>

```rust
fn main(){
  let a = Box::new(5); //5是在堆上，堆的地址在栈上. a具有 栈上 和 堆上值的所有权。
  assert_eq!(5, *a);
  let mut b = Box::new(5);
  *b = 100;
  assert_eq!(5, *a); // mut b 表示既可以改变 b 绑定的栈上的值，也可以改变 堆上的值。
}
```

# Rc<T>
> 带引用计数的智能指针。共享所有权。

```rust
enum List {
    Cons(i32, Rc<List>),
    Nil,
}

use crate::List::{Cons, Nil};
use std::rc::Rc;

fn main() {
    let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
    let b = Cons(3, Rc::clone(&a));
    let c = Cons(4, Rc::clone(&a));
}
```

# RefCell<T>
功能：即使 `RefCell<T>` 不是 mut 的，也可以改变 `T` 的值
```rust
use std::cell::RefCell;
fn main(){
  let a = RefCell<Vec<String>>;
  a.borrow_mut().push(String::from("hello"));
}
```

