内存安全问题：
* 错误的内存访问：读取未初始化内存；解引用空指针；缓存溢出
* 违反生命周期：invalid free；use after free; double free

线程安全问题：
* 死锁；data race

# rust所有权
> rust 的基础规则是什么？解决了什么问题？如何解决的？

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
	* 编译期就能避免 并行编程时碰到的data race问题。当以下3个行为同时发生时，就会出现data race：1）多个指针同时访问一块内存，2）至少有一个指针用来写这块内存，3）并没有机制来同步这块内存的访问
	* 该机制能够正确的保证 栈内存 不会出现 data race问题
	* 当然：该规则还会有一些其它的诡异的影响。。。。。 
* References must always be valid.
	* 防止 dangling pointer

rust 使用什么机制保证我们遵守该规则
* At any given time, you can have either one mutable reference or any number of immutable references.
	* 没仔细研究，不知道
* References must always be valid.
	* 使用生命周期标注。需要我们 **正确的进行** 生命周期标注

## 引用计数
* 大多数情况下，单一ownership规则都可以work。但是总有些情况，一块内存是有多个 owner 的。这时候使用 `Rc<T>` 来管理！！！
* 我们无法对 **栈空间** 具有多个ownership
我们看一下如何使用

```rust
fn main(){
    let a = String::from("hello");
    let rc = Rc::new(a); // 这里是由ownership转移的，a的ownership已经放弃了。a管理的堆内存已经交给 rc 来管理了
    
    let a = 10;
    let rc = Rc::new(a); // 这里new，就直接在堆上分配一个空间用来存值了。还是共享的堆空间了。 通过rc，是无法拿到raw pointer的。
    println!("{}", rc);
}
```


**DerefMut** 的存在，阻止了好多有问题的代码！！！

# rust通过哪些机制保证了 内存安全&并发安全
https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-798.pdf


内存安全问题：
* 错误的内存访问：
	* 读取未初始化内存；rust保证内存的初始化？
	* 解引用空指针；
	* 缓存溢出; 
* 违反生命周期：使用ownership修理了
	* invalid free；
	* use after free; 
	* double free

线程安全问题：
* 死锁；没有解决
* data race；

上述的 引用&借用规则，仅仅是控制住了

## 并发安全
* 死锁：并没有解决
* data race：
	1. data race：多个线程共同读写同一块内存，但是没有对该内存没有同步机制。
	2. 并发条件下 rust 根本无法共享栈上的 内存。根本构造不出来。只能共享堆上的数据
	3. 并发过程中的共享只能使用 `std::sync::Arc`, 但是仅仅 `Arc` 还不行，`Arc` 只能读，不能写，因为没有实现 `DerefMut trait`
	4. 想到可变，就想到了 `std::cell::RefCell`，但是用 `Arc<RefCell>` 也会报错。**这个原因是啥？如何保证的**
	5. 最终，只有一条路：`std::sync::Mutex`, 使用 `Arc<Mutex<T>>` 就ok了。而且这种形式，也保证了 data race 不会发生。
	6. 总结：`Arc` 保证了多线程情况下，引用计数能够正确的加减。`Mutex` 提供了同步机制 & 内部可变性机制。`Arc<Mutex<T>>` 等价于单线程情况下的 `Rc<RefCell<T>>` 


# 生命周期标注

> rust 中的生命周期标注主要是解决悬垂指针问题的。生命周期标注可以和泛型对照去理解


https://hashrust.com/blog/lifetimes-in-rust/

https://doc.rust-lang.org/stable/book/ch10-03-lifetime-syntax.html

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {

}
```

生命周期标注是用来**关联** 一个函数中的不同参数及返回值的生命周期的。rust会根据这些标注的关系对代码进行检查
```rust
// 1. 如果生命周期约束与实际方法不符，会报错
// 编译器会报错。因为明明标注返回的生命周期是 'a, 但实际返回的却是 'b
fn longest<'a, 'b>(x: &'a str, y: &'b str) -> &'a str {
 	y
}


```


如何立即上面的标注是重点：上述的标注指定了输入参数 & 返回值 生命周期关系的约束
	1. 对于某个生命周期 `'a`，传入参数的生命周期必然不能短于该生命周期。
	2. 返回的引用必须和函数的参数具有相同的生命周期/返回的引用必然需要不小于`'a`的生命周期
	3. 返回接收值的 生命周期一定要小于等于 `'a`

生命周期 `'a` 是如何确定的呢？编译器在编译函数调用的时候，会将 `'a` 指定为 `x/y` 中生命周期比较短的那个


### 结构体中的生命周期标注

```rust

// 该标注的含义是：该结构体实例的存活时间不能超过 part 引用的存活时间
struct ImportantExcerpt<'a> {
	part: &'a str,
}

// 结构体字段中的生命周期放在这里
impl<'a> ImportantExcerpt<'a> {

}
```


### 静态生命周期

`'static` ，表示整个程序的执行期，所有的字符串常量都是 `'static`

```rust
let s : &'static str = "hello world";
let s2 = "hello world"; // same
```

### more

!!!! 参数生命周期的大小关系无法代表 泛型生命周期参数的生命周期大小关系

```rust
// 'b: 'a 说明泛型生命周期参数 'b 要长于 'a。注意不是 p2 生命周期长于 p1
fn demo1<'a, 'b: 'a>(p1: &'a str, p2: &'b str) -> &'a str {
    p1
}

 fn main() {
        let p1 =  String::from("world");
        // let mut c;
        {
            let p2 = String::from("hello");

			// 这里 p2 生命周期长于 p1. 但是泛型生命周期参数 'b >= 'a 所以，编译器会将 'b, 'a 都推断成与 p1 生命周期一致
            let c = demo1(p1.as_str(), p2.as_str());
            println!("{}", c);

        }
    }

```