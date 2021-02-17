# 无谓并发
Here are the topics we’ll cover in this chapter:

* How to create threads to run multiple pieces of code at the same time
* Message-passing concurrency, where channels send messages between threads
* Shared-state concurrency, where multiple threads have access to some piece of data
* The Sync and Send traits, which extend Rust’s concurrency guarantees to user-defined types as well as types provided by the standard library

## thread::spawn
> 通过thread::spawn来创建线程

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // spawn之后线程就开始跑了。
    thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }
}
```

> 使用 join 等待线程执行完毕

```rust
use std::thread;
use std::time::Duration;

fn main() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();
}
```

> 线程中使用 move closures
```rust
// 错误代码
use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    // 根据 闭包的捕获规则（是啥？）v会被 引用捕获。因为会新开线程，并不知道在当前线程 v后序会不会被搞，所以这里是会报错的
    let handle = thread::spawn(|| {
        println!("Here's a vector: {:?}", v);
    });

    handle.join().unwrap();
}
```
```rust
use std::thread;

fn main() {
    let v = vec![1, 2, 3];
    // 使用 move closure。直接发生所有权的移交。
    let handle = thread::spawn(move || {
        println!("Here's a vector: {:?}", v);
    });

    handle.join().unwrap();
}
```

## 线程之间的数据传输
这部分和golang的channel相似。`Do not communicate by sharing memory; instead, share memory by communicating.”`
* rust实现的是 `multiple producer, single consumer` 的 channel
* rx.send 的时候，会将 所有权也 send出去！
* 
```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    println!("spawning thread");
    thread::spawn(move || {
        let val = String::from("hi");
        thread::sleep(Duration::from_secs(3));
        tx.send(val).unwrap();
    });
    println!("spawning thread done");

    let received = rx.recv().unwrap(); //阻塞接收。也可以使用 try_recv(非阻塞！)
    println!("Got: {}", received);
}
```

* 构建 multiple producer
```rust
let (tx, rx) = mpsc::channel();

let tx1 = mpsc::Sender::clone(&tx); //构建 multiple producer 的核心
thread::spawn(move || {
    let vals = vec![
        String::from("hi"),
        String::from("from"),
        String::from("the"),
        String::from("thread"),
    ];

    for val in vals {
        tx1.send(val).unwrap();
        thread::sleep(Duration::from_secs(1));
    }
});

thread::spawn(move || {
    let vals = vec![
        String::from("more"),
        String::from("messages"),
        String::from("for"),
        String::from("you"),
    ];

    for val in vals {
        tx.send(val).unwrap();
        thread::sleep(Duration::from_secs(1));
    }
});

for received in rx {
    println!("Got: {}", received);
}
```

## 数据共享
> 上述的 channel 是数据共享的一种方式。现在介绍另一种方式。

* mutex：和 c++, go的mutex 不同。rust的mutex是个模板，还能存值？？？
```rust
use std::sync::Mutex;

fn main() {
    let m = Mutex::new(5);

    {
        let mut num = m.lock().unwrap(); // unwrap 得到的实际上是一个 MutexGuard. 可以通过该值操作 mutex里面的值。该 对象出了 scope 还会自动 unlock
        *num = 6;
    }

    println!("m = {:?}", m);
}
```

* 多线程如何共享 Mutex?
通过智能指针`Rc<T>`，可以使得一个值拥有多个owner。但是`Rc<T>` 无法用在多线程场景下。在多线程场景下，使用`Arc<T>` 来代替 `Rc<T>`
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0)); // Mutex<T> provides interior mutability
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || { //这里依旧是 move closure。
            let mut num = counter.lock().unwrap();

            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

# async

# tokio

* Executor执行 task
* task是 tokio.spawn的`async block`
* `async fn` 里面才能执行 `.await`

```rust
#[tokio::main]
async fn main() {
    println!("hello");
}
```
会被转换成以下代码
```rust
fn main() {
    let mut rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        println!("hello");
    })
}

```
