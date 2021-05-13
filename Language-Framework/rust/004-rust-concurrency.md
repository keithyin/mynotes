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

* 多线程如何共享 `Mutex` ?
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

## `Sync & Send` marker trait

> `send` & `sync` 是 marker trait，什么是 `marker trait` 呢？就是编译器负责做标记，rust用户无法干预的 trait。



`Send`: 如果一个类被标记了 `Send`，那么可以在**线程间传递其所有权**。

* 只有被标记了 `Send` 的类的对象在 线程间传递所有权时，编译器才不会报错。
* `Send` 标记保证了对象线程间传递所有权的正确性
* 所以在判断一个 类是不是 `Send` 的时候，就需要考虑其 在线程间传递所有权会不会有什么问题
  * 比如：`Rc<T>`. 线程间传递所有权就会导致 `reference count` 计算出现问题，所以就不是 `Send`
  * 但是像 `i32, char ...` 这些的，线程间移动也没得什么问题。因为不涉及什么需要同步的数据的修改。

一个常见的线程间传递所有权的场景为(和多线程共享 `Mutex` 使用同一份代码解释)：

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0)); // Mutex<T> provides interior mutability
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
      	// main线程 counter 的所有权 移交给了 spawn 的线程
        let handle = thread::spawn(move || { 
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

> rust中，几乎所有类型都是 `Send` 。 但是有一些特殊情况，比如 `Rc<T>` 就不是 `Send`，这是因为当我们 `clone` 一个 `Rc<T>` 将其移动到另一个线程的时候，两个线程有可能同时修改其 `reference count`，这会导致 `data race`。 
>
> 所以，目前来看，包含 引用计数的类，如果想 `Send` 的话，那么其 引用计数的修改必须要同步的，比如 `Arc<T>`.
>
> `Send` 的标记规则：如果一个类的所有字段是 `Send`, 那么该类就是 `Send`. Almost all primitive types are `Send`, aside from raw pointers,



`Sync`: 如果一个类被标记了 `Sync` , 那么该类的对象可以被多个线程同时 引用( `Reference`). 比如：`Mutex<T>` 就可以被多个线程同时引用(`Arc<Mutex<T>>`) 。

* `Sync` 表示的是一个对象是否可以被多个线程同时访问。`Mutex<T>` 就可以，`Cell<T>, RefCell<T>` 就不行，他俩多线程访问，可能会出现 `data race` 问题。

* 如果 `Arc<T>` 是 `Send` ，那么 `T` 是 `Sync`。

* 


所以多线程共享变量的基本流程为：

1. 创建一个 `Sync` 对象，（想修改的话用 `Mutex`（不要用 `Cell, RefCell`）, 如果不想修改用原始值就OK了）
2. 然后将其封装到 `Arc` 中构建出一个 `Send`. 
3. 然后将其移动到不同的线程中。
4. 多线程就可以共享 `Sync` 对象了。



编码时常见问题解决方式：

* 如果编译器报错某 对象 不是 `Sync` ，可以套一个 `Mutex` 解决。
* 如果编译器报错 某对象不是 `Send`, 那就换一个 `Send` 的对象吧。



# async

* `async block` : 实际是一个 `Future object`

* `async fn`: 调用返回的是一个 `Future object`

`Future object` 包含了需要进行的操作，异步的调度器实际就是在调度 `Future object`。



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
## spawning

```rust
use tokio::net::TcpListener;
use tokio::net::{TcpListener, TcpStream};
use mini_redis::{Connection, Frame};

async fn process(socket: TcpStream) {
    // The `Connection` lets us read/write redis **frames** instead of
    // byte streams. The `Connection` type is defined by mini-redis.
    let mut connection = Connection::new(socket);

    if let Some(frame) = connection.read_frame().await.unwrap() {
        println!("GOT: {:?}", frame);

        // Respond with an error
        let response = Frame::Error("unimplemented".to_string());
        connection.write_frame(&response).await.unwrap();
    }
}

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:6379").await.unwrap();

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        // A new task is spawned for each inbound socket. The socket is
        // moved to the new task and processed there.
        // 新开一个 task 处理 socket
        tokio::spawn(async move {
            process(socket).await;
        });
    }
}
```

**tasks**
tokio 的task 是一个 异步green thread. 通过 将一个 `async block` 传给 `tokio.spawn` 来创建。`tokio.spawn` 返回一个 `JoinHandle`，调用者可以通过这个 `JoinHandle` 来和 `spawned task` 进行交互。`async block` 可以返回值，调用者可以通过在 `JoinHandle` 上调用 `await` 来获取返回值。

`tasks` 是 tokio scheduler 管理的最小执行单元。` Spawning the task` 会将 `task` 提交给 `Tokio scheduler`, 由`Tokio scheduler`决定该如何调度。`spawned task` 可能在 `spawn` 它的 线程上执行，也可能在不同的线程上执行。`spawned task` 可以在 不同的线程之间来回移动。 

```rust
#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        // Do some async work
        "return value"
    });

    // Do some other work

    let out = handle.await.unwrap();
    println!("GOT {}", out);
}
```
`JoinHandle.await`返回的是`Result`，当task在执行的时候碰到错误(task panic or task is forcefully cancelled by the runtime shutting down)，`JoinHandle.await`就会返回`Err`. 

**task必须满足的几个条件**

* `'static` bound
* `Send` bound

`'static` ：task 的 type是`'static` 的含义是：`async block`中不能含有对外部变量的的引用。

> `'static` 并不意味着 lives forever
```rust
use tokio::task;

#[tokio::main]
async fn main() {
    let v = vec![1, 2, 3];

    task::spawn(async {
        println!("Here's a vec: {:?}", v); //这里的 v 是 外部变量的引用。编译会报错
    });
}
```

```rust
use tokio::task;

#[tokio::main]
async fn main() {
    let v = vec![1, 2, 3];

    task::spawn(async move{
        println!("Here's a vec: {:?}", v); // 这里就是 owner 了
    });
}
```

`Send`: 因为 `task` 中一旦 `.await` 后，该 `task` 就可能被 `tokio scheduler` 在不同的线程中移来移去，所以生存周期跨越了 `.await` 的变量，必须实现了 `Send trait`

```rust
// 这个不work，因为 std::rc::Rc 没有实现 Send
use tokio::task::yield_now;
use std::rc::Rc;

#[tokio::main]
async fn main() {
    tokio::spawn(async {
        let rc = Rc::new("hello");

        // `rc` is used after `.await`. It must be persisted to
        // the task's state.
        yield_now().await;

        println!("{}", rc);
    });
}
```

```rust
// 这个可以work，因为 std::rc::Rc 的生命周期没有跨越 .await
use tokio::task::yield_now;
use std::rc::Rc;

#[tokio::main]
async fn main() {
    tokio::spawn(async {
        // The scope forces `rc` to drop before `.await`.
        {
            let rc = Rc::new("hello");
            println!("{}", rc);
        }

        // `rc` is no longer used. It is **not** persisted when
        // the task yields to the scheduler
        yield_now().await;
    });
}
```



## Shared state

* 使用 `Mutex`

```rust
use bytes::Bytes;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Db = Arc<Mutex<HashMap<String, Bytes>>>;

use tokio::net::TcpListener;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() {
    let listener = TcpListener::bind("127.0.0.1:6379").await.unwrap();

    println!("Listening");

    let db = Arc::new(Mutex::new(HashMap::new()));

    loop {
        let (socket, _) = listener.accept().await.unwrap();
        // Clone the handle to the hash map.
        let db = db.clone();

        println!("Accepted");
        tokio::spawn(async move {
            process(socket, db).await;
        });
    }
}

use tokio::net::TcpStream;
use mini_redis::{Connection, Frame};

async fn process(socket: TcpStream, db: Db) {
    use mini_redis::Command::{self, Get, Set};

    // Connection, provided by `mini-redis`, handles parsing frames from
    // the socket
    let mut connection = Connection::new(socket);

    while let Some(frame) = connection.read_frame().await.unwrap() {
        let response = match Command::from_frame(frame).unwrap() {
            Set(cmd) => {
                let mut db = db.lock().unwrap();
                db.insert(cmd.key().to_string(), cmd.value().clone());
                Frame::Simple("OK".to_string())
            }           
            Get(cmd) => {
                let db = db.lock().unwrap();
                if let Some(value) = db.get(cmd.key()) {
                    Frame::Bulk(value.clone())
                } else {
                    Frame::Null
                }
            }
            cmd => panic!("unimplemented {:?}", cmd),
        };

        // Write the response to the client
        connection.write_frame(&response).await.unwrap();
    }
}
```



## Channels

tokio实现了好多`channel`

* `oneshot`: 单生产者，单消费者。 A single value can be sent.
* `watch`: 单生产者，多消费者. Many values can be sent, but no history is kept. Receivers only see the most recent value.
* `mpsc`: 多生产者，单消费者. Many values can be sent.
* `broadcast`: 多生产者，多消费者. Many values can be sent. Each receiver sees every value.

```rust
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);
    let tx2 = tx.clone();

    tokio::spawn(async move {
        tx.send("sending from first handle").await;
    });

    tokio::spawn(async move {
        tx2.send("sending from second handle").await;
    });

    while let Some(message) = rx.recv().await {
        println!("GOT = {}", message);
    }
}
```

```rust
use tokio::sync::oneshot;
use bytes::Bytes;

/// Multiple different commands are multiplexed over a single channel.
#[derive(Debug)]
enum Command {
    Get {
        key: String,
        resp: Responder<Option<Bytes>>,
    },
    Set {
        key: String,
        val: Vec<u8>,
        resp: Responder<()>,
    },
}

/// Provided by the requester and used by the manager task to send
/// the command response back to the requester.
type Responder<T> = oneshot::Sender<mini_redis::Result<T>>;

let t1 = tokio::spawn(async move {
    let (resp_tx, resp_rx) = oneshot::channel();
    let cmd = Command::Get {
        key: "hello".to_string(),
        resp: resp_tx,
    };

    // Send the GET request
    tx.send(cmd).await.unwrap();

    // Await the response
    let res = resp_rx.await;
    println!("GOT = {:?}", res);
});

let t2 = tokio::spawn(async move {
    let (resp_tx, resp_rx) = oneshot::channel();
    let cmd = Command::Set {
        key: "foo".to_string(),
        val: b"bar".to_vec(),
        resp: resp_tx,
    };

    // Send the SET request
    tx2.send(cmd).await.unwrap();

    // Await the response
    let res = resp_rx.await;
    println!("GOT = {:?}", res);
});
```


## I/O

`async fn read()`
* `AsyncReadExt::read` 提供了异步的方式来将数据读入`buffer`, 它会返回读入的字节个数
* `AsyncReadExt::read_to_end` 读文件中的数据，直到碰到EOF。

```rust
use tokio::fs::File;
use tokio::io::{self, AsyncReadExt};

#[tokio::main]
async fn main() -> io::Result<()> {
    let mut f = File::open("foo.txt").await?;
    let mut buffer = [0; 10];

    // read up to 10 bytes
    let n = f.read(&mut buffer[..]).await?;

    println!("The bytes: {:?}", &buffer[..n]);
    
    let mut f = File::open("foo.txt").await?;
    let mut buffer = Vec::new();

    // read the whole file
    f.read_to_end(&mut buffer).await?;
    Ok(())
}
```

* `AsyncWriteExt::write` 将buffer中的数据写入writer，返回写入的字节数。
* `AsyncWriteExt::write_all` 将buffer中所有的数据写入writer

```rust
use tokio::io::{self, AsyncWriteExt};
use tokio::fs::File;

#[tokio::main]
async fn main() -> io::Result<()> {
    let mut file = File::create("foo.txt").await?;

    // Writes some prefix of the byte string, but not necessarily all of it.
    let n = file.write(b"some bytes").await?;

    println!("Wrote the first {} bytes of 'some bytes'.", n);
    
    let mut buffer = File::create("foo.txt").await?;

    buffer.write_all(b"some bytes").await?;
    Ok(())
}
```

**helper functions**

**echo server**

## Framing

> framing is the process of taking a byte stream and converting it to a stream of frames. A frame is a unit of data transmitted between two peers.


## select



## mini_tokio

* Executor 调度 task
* task 封装了 future & sender
  * future 代表要执行的操作
  * sender 负责当 wake 的时候，将自己再 send 给 executor 的调度队列
    * 该逻辑应该 `impl ArcWake for Task {fn wake_by_ref()}` 中实现。
  * task 即是 要执行的任务，也是 waker！自己唤醒自己。

