# rust范型

## Generic Data Types
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

## trait（函数集合）
> trait 函数集合。struct 属性集合

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}
```

* 为Type实现Trait
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

* trait 作为形参
```rust
// 所有实现了 Summary 的 对象都可以传进去
pub fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

* Trait bound syntex: 上述的 `&impl Summary`这种做法实际是 `Trait bound syntext` 的 `syntex sugar`
```rust
pub fn notify<T: Summary>(item: &T) {
    println!("Breaking news! {}", item.summarize());
}
```

```rust
// 以下两种语法等价
pub fn notify(item1: &impl Summary, item2: &impl Summary) {}
pub fn notify<T: Summary>(item1: &T, item2: &T) {}
```

* 如果有多个 Trait bound, 怎么写呢
```rust
pub fn notify(item: &(impl Summary + Display)) {}
pub fn notify<T: Summary + Display>(item: &T) {}

// 使用 where clause 使得语法更清晰
fn some_function<T: Display + Clone, U: Clone + Debug>(t: &T, u: &U) -> i32 {}
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{}
```

* Using Trait Bounds to Conditionally Implement Methods
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
