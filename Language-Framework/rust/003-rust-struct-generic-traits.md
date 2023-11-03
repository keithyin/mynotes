# 结构体 & 方法

```rust
struct Value {

}

impl Value {
  // 如果第一个形参 不是 self，那么就类似于 c++ 的静态方法。
  pub fn new() -> Self{Value{}}

  // 方法。第一个形参有多种选择
  // `self`, `&self`, `&mut self`, `self: Box<Self>`, `self: Rc<Self>`, `self: Arc<Self>`, or `self: Pin<P>` (where P is one of the previous types except `Self`)
  // self: &Rc<Self> 也行，应该意味着 `self: &Box<Self>`, `self: &Arc<Self>` 也👌
  pub fn do_something(&self) {

  }

  // Box::new(Value::new()).do_something2();
  pub fn do_something2(self: Box<Self>) {

  }
}
```

# Generic Data Types

```rust
struct Point<T> {
    x: T,
    y: T,
}

// impl<T> 这个T用来说明 T 是范型，而非具体类型！
impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// 只为 Point<f32> 实现方法
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn main() {
    let integer = Point { x: 5, y: 10 };
    let float = Point { x: 1.0, y: 4.0 };
}
```

## 自动类型推断

rust 可以通过1）实参，2）返回值 进行自动类型推断，当然我们也可以显式指定泛型类型

```rust
// 通过该函数签名可以看出：无法通过实参进行类型自动推断，
// 所以只能显式指定，或者利用返回值进行自动推断
// let a = demo_init::<TypeA>(10); 显式指定
// let a: TypeA = demo_init(10); 返回值类型自动推断

fn<T: Init> demo_init(i: i32) -> T {
    T::init(0)
}

// 通过函数签名就可以 自动类型推断
fn<T: Mul> demo_mul(val: T) -> T {
    val.mul(3)
}

//

impl<T> for TypeA {
    // 这个方法就只能通过返回值 进行自动类型推断。 甚至无法手动指定。
    fn new(val: i32) -> Self {
        T::do_some()
    }    
}


```



# trait（函数集合）

> trait 函数集合。struct 属性集合

```rust
pub trait Summary {
    // 因为 trait 是需要 struct 实现的。所以第一个参数可能是 &self, &mut self, self?
    fn summarize(&self) -> String;
    // 因为 trait 是需要 struct 实现的。 Self 表示调用该方法的 struct
    fn demo(&self) -> Self;
}
```

## 为 Struct 实现 Trait

```rust
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

## 为泛型Struct实现Trait
```rust
// 泛型结构体
pub struct CommonNode <T> {
    num_threads: i32,
    pre_receiver: Option<Receiver<T>>,

    cur_sender: Option<Sender<T>>,
    cur_receiver: Option<Receiver<T>>,

    work_func: Option<fn (Option<Receiver<T>>, Option<Sender<T>>)>,

}

impl<T: Send + 'static> CommonNode<T> {

    pub fn new(num_threads: i32, work_func: fn (Option<Receiver<T>>, Option<Sender<T>>)) -> Self {
        let (s, r) = channel::unbounded::<T>();
        CommonNode { num_threads: num_threads,
            pre_receiver: None, 
            cur_sender: Some(s), 
            cur_receiver: Some(r),
            work_func: Some(work_func)
         }
    }
}

// 泛型结构体实现Trait。如果T 有 Trait的约束的话这么写
// 如果没有泛型约束的话 impl<T> PipelineNode for CommonNode<T> 即可
impl<T: Send + 'static> PipelineNode for CommonNode<T>  {
    type CommType = T;

    fn get_cur_receiver(&mut self) -> Receiver<Self::CommType> {
        self.cur_receiver.take().unwrap()
    }

    fn get_cur_sender(&mut self) -> Sender<Self::CommType> {
        self.cur_sender.take().unwrap()
    }

    fn set_pre_receiver(&mut self, receiver: Receiver<Self::CommType>) {
        self.pre_receiver = Some(receiver);
    }

}

```

## trait 作为形参

[Why does `dyn Trait` require a Box? - help - The Rust Programming Language Forum](https://users.rust-lang.org/t/why-does-dyn-trait-require-a-box/23471)

[Difference between returning dyn Box&lt;Trait&gt; and impl Trait - #3 by H2CO3 - The Rust Programming Language Forum](https://users.rust-lang.org/t/difference-between-returning-dyn-box-trait-and-impl-trait/57640/3)

```rust
// 所有实现了 Summary 的 对象都可以传进去（传的是引用）
// 静态派发
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}

// 所有实现了 Summary 的 对象都可以传进去（传的是对象）
pub fn notify(item: impl Summary) {
    println!("Breaking news! {}", item.summarize());
}


// 动态派发！ (dyn Summary 是 Unsized 的，不能直接搞这么一个类型，只能通过引用。)
// 调用时传对象引用
pub fn notify(item: &dyn Summary) {
    println!("Breaking news! {}", item.summarize());
}
pub fn notify(item: Box<dyn Summary>) {
    println!("Breaking news! {}", item.summarize());
}
// 编译会报错 dyn是用来描述对象大小不确定的？
pub fn notify(item: dyn Summary) {
    println!("Breaking news! {}", item.summarize());
}
```



## Trait bound syntex （限制泛型对象的 类型范围。即：只有实现了trait的类型对象可以作为实参）

上述的 `&impl Summary`这种做法实际是 `Trait bound syntext` 的 `syntax sugar`

```rust
// 静态派发。编译器会为调用生成特定代码。
// T:Summary, trait 约束。
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

```rust
// 以下两种语法等价。
pub fn notify(item1: &impl Summary, item2: &impl Summary) {}
pub fn notify<T: Summary>(item1: &T, item2: &T) {}
```

## 如果有多个 Trait bound, 怎么写呢

```rust
pub fn notify(item: &dyn Summary + Display) {}

pub fn notify(item: &(impl Summary + Display)) {}
pub fn notify(item: impl Summary + Display) {}

pub fn notify<T: Summary + Display>(item: &T) {}

// 使用 where clause 使得语法更清晰
fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {}
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{}
```

## Using Trait Bounds to Conditionally Implement Methods

```rust
use std::fmt::Display;

struct Pair<T> {
    x: T,
    y: T,
}

impl<T> Pair<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

// 只为 实现了 Display + PartialOrd trait 的 T 实现 cmp_display 方法。
impl<T: Display + PartialOrd> Pair<T> {
    fn cmp_display(&self) {
        if self.x >= self.y {
            println!("The largest member is x = {}", self.x);
        } else {
            println!("The largest member is y = {}", self.y);
        }
    }
}
```

* Using Trait Objects That Allow for Values of Different Types. `Box<dyn Draw>`:which is a trait object; it’s a stand-in for any type inside a Box that implements the Draw trait!!
  
  ```rust
  pub struct Screen {
    pub components: Vec<Box<dyn Draw>>,
  }
  ```

## trait 的 type placeholder

> *Associated types* connect a type placeholder with a trait such that the trait method definitions can use these placeholder types in their signatures. The implementor of a trait will specify the concrete type to be used in this type’s place for the particular implementation. That way, we can define a trait that uses some types without needing to know exactly what those types are until the trait is implemented.

如果 trait 的方法是个 范型方法，应该怎么写呢？https://doc.rust-lang.org/book/ch19-03-advanced-traits.html

* 使用 `type placeholder`
* 使用 范型

```rust
// type placeholder
pub trait Iterator {
    type Item; // 类型的 placeholder。 impl Trait for Struct 时候填入。

    fn next(&mut self) -> Option<Self::Item>; // self 表示 Struct 对象，Self表示 Struct 类。
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        // --snip--
    }
}
```

```rust
// 使用范型
struct Counter {
    data: Vec<i32>,
    cur_pos: usize,
}

impl Counter {
    pub fn new(data: Vec<i32>) -> Self{
        Counter{data, cur_pos: 0}
    }
}

trait Iterator<T> {
    fn next(&mut self) -> Option<T>;
}

impl Iterator<i32> for Counter {
    fn next(&mut self) -> Option<i32> {
        if self.cur_pos >= self.data.len() {
            return None;
        }
        let value = self.data[self.cur_pos];
        self.cur_pos += 1;
        Some(value)
    }
}

impl Iterator<u32> for Counter {
    fn next(&mut self) -> Option<u32> {
        if self.cur_pos >= self.data.len() {
            return None;
        }
        let value = self.data[self.cur_pos];
        self.cur_pos += 1;
        Some(value as u32)
    }
}

#[cfg(test)]
mod test {
    use crate::trait_demo::{Counter, Iterator};

    #[test]
    fn test_trait() {
        let mut counter = Counter::new(vec![1, 2, 3, 4, 5]);
        let val: Option<i32> = counter.next();
    }
}
```


## 带Type placehoulder的trait作为形参

```rust
// 所有实现了 Summary 的 对象都可以传进去（传的是引用）
// 静态派发
pub fn notify<T> (item: &impl Summary<TypePlaceholder=T>) {
    println!("Breaking news! {}", item.summarize());
}

// 所有实现了 Summary 的 对象都可以传进去（传的是对象）
pub fn notify<T> (item: impl Summary<TypePlaceholder=T>) {
    println!("Breaking news! {}", item.summarize());
}


// 动态派发！ (dyn Summary 是 Unsized 的，不能直接搞这么一个类型，只能通过引用。)
// 调用时传对象引用
pub fn notify<T>(item: &dyn Summary<TypePlaceholder=T>) {
    println!("Breaking news! {}", item.summarize());
}
pub fn notify<T>(item: Box<dyn Summary<TypePlaceholder=T>>) {
    println!("Breaking news! {}", item.summarize());
}
// 编译会报错. dyn是用来描述对象大小不确定的？
pub fn notify(item: dyn Summary<TypePlaceholder=T>) {
    println!("Breaking news! {}", item.summarize());
}
```

## Newtype Pattern

在开发中，如果想对 一个 `struct` 实现一个 `trait` 的话，要么 `struct` 在我们的 crate中，要么 `trait` 在我们的 `crate` 中。但是我们可以通过Newtype Pattern来规避这个限制。

Newtype Pattern：使用 Tuple Structs without named fields to create different types

```rust
use std::fmt;

struct Wrapper(Vec<String>);

impl fmt::Display for Wrapper {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "[{}]", self.0.join(", "));
  }
}
```
