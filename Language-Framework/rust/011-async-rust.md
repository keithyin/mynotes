rust的异步编程模型

* 异步方法：`async fn` ，调用一个 `async fn` 实际返回一个 `Future`. 这时实际上是没有什么计算的。

* 异步block：`async {}`, 该块会返回一个 `Future object` ，这时也没有啥计算。

  

* 如果要真实计算的话，需要在 `Executor` 上执行 `Future`

* 关于 `Future` 的执行，有两种不同情况

  * 在 `async fn` 中如何执行 `Future`
    * `.await` 
  * 在 普通 `fn` 中如何执行 `Future`

* `Executor` 实际上就是在调度 `Future` 的执行

* `Future` 是异步编程的核心，`Future` 是一个可以产生值的 异步计算。

  * 

  * ```rust
    trait SimpleFuture {
        type Output;
        fn poll(&mut self, wake: fn()) -> Poll<Self::Output>;
    }
    
    enum Poll<T> {
        Ready(T),
        Pending,
    }
    ```

  * `Future` 通过调用 `poll` 执行计算。如果计算完成，返回 `Pool::Ready(result)` . 如果此次计算未完成，那就返回 `Pool::Pending` 即可。

  * `wake`: 如果 `Future` 返回 `Pool::Pending`, 那么该`Future` 就扔到未就绪队列了，如何将其再放到就绪队列中呢？就需要这个 `wake` 了，他可以是一个 `Timer`, 也可以是其它的一些东西。

  * 一个简单的 `SocketRead` 代码可以如下所示

    * ```rust
      pub struct SocketRead<'a> {
          socket: &'a Socket,
      }
      
      impl SimpleFuture for SocketRead<'_> {
          type Output = Vec<u8>;
      
          fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
              if self.socket.has_data_to_read() {
                  // The socket has data-- read it into a buffer and return it.
                  Poll::Ready(self.socket.read_buf())
              } else {
                  // The socket does not yet have data.
                  //
                  // Arrange for `wake` to be called once data is available.
                  // When data becomes available, `wake` will be called, and the
                  // user of this `Future` will know to call `poll` again and
                  // receive data.
                  self.socket.set_readable_callback(wake); // 回调代码被封装起来了。
                  Poll::Pending
              }
          }
      }
      ```

* 





# Why Async

考虑一个任务，“并发的下载两个网页”，使用 Thread 解决方案的话，我们会这么组织代码
```rust
fn get_two_sites() {
    // Spawn two threads to do work.
    let thread_one = thread::spawn(|| download("https://www.foo.com"));
    let thread_two = thread::spawn(|| download("https://www.bar.com"));

    // Wait for both threads to complete.
    thread_one.join().expect("thread one panicked");
    thread_two.join().expect("thread two panicked");
}
```
使用多线程会有两个问题：
* 线程之间的切换会有开销
* 线程之间的数据共享也会有开销（同步问题）

以上两个问题正式 `async` 代码想要解决的。`rust` 通过 `async/.await` 关键字来提供 `async` 编程支持。
```rust
async fn get_two_sites_async() {
    // Create two different "futures" which, when run to completion,
    // will asynchronously download the webpages.
    let future_one = download_async("https://www.foo.com");
    let future_two = download_async("https://www.bar.com");

    // Run both futures to completion at the same time.
    join!(future_one, future_two);
}
```

`async` 和 `thread` 有以下几点不同：
* async没有线程切换带来的消耗
* thread 是由操作系统来进行切换的，async代码的协程切换需要用户自己写代码
* 一个线程 会有 多个 协程在之上运行？

在Rust中`async fn` 创建了一个 `asynchronous function`, 该函数返回一个 `Future`. 执行返回值的时候，才是真正的执行函数体！

# 如何使用 async & await
1. 添加依赖
```rust
[dependencies]
futures = "0.3"
```
2. 使用 `async fn` 创建一个 `asynchronous function`, 该函数会返回一个 `Future`, 如果想让 函数执行的话，这个 `Future` 需要在一个 `executor` 中执行
```rust
// `block_on` blocks the current thread until the provided future has run to
// completion. Other executors provide more complex behavior, like scheduling
// multiple futures onto the same thread.
use futures::executor::block_on;

async fn hello_world() {
    println!("hello, world!");
}

fn main() {
    let future = hello_world(); // Nothing is printed
    block_on(future); // `future` is run and "hello, world!" is printed, block_on 会阻塞当前进程，直到 future 执行完成。
}
```
3. 在 `async fn` 中，可以使用 `.await` 来等待 `Future` 完成， 和`block_on()` 不同 `.await` 并不阻塞当前线程，将当前线程的控制权交出，异步的等待 `future` 完成，

```rust
// 唱歌 & 跳舞 是可以并发执行的，但是 学习唱歌 & 唱歌 是序列执行的。所以代码可以写成以下方式。
async fn learn_song() -> Song { /* ... */ }
async fn sing_song(song: Song) { /* ... */ }
async fn dance() { /* ... */ }

async fn learn_and_sing() {
    // Wait until the song has been learned before singing it.
    // We use `.await` here rather than `block_on` to prevent blocking the
    // thread, which makes it possible to `dance` at the same time.
    let song = learn_song().await; // 阻塞， 然后将当前线程的控制权交给其它人。
    sing_song(song).await;
}

async fn async_main() {
    let f1 = learn_and_sing();
    let f2 = dance();

    // `join!` is like `.await` but can wait for multiple futures concurrently.
    // If we're temporarily blocked in the `learn_and_sing` future, the `dance`
    // future will take over the current thread. If `dance` becomes blocked,
    // `learn_and_sing` can take back over. If both futures are blocked, then
    // `async_main` is blocked and will yield to the executor.
    futures::join!(f1, f2);
}

fn main() {
    block_on(async_main());
}
```

# Future 到底是什么
`Future` 是 `rust` 异步编程的核心，它表示了一个 计算某个值的 异步计算，当然，这个值也可以是 `()`。一个简单的 `Future` 可以是下面这个样子
```rust

trait SimpleFuture {
    type Output;
    fn poll(&mut self, wake: fn()) -> Poll<Self::Output>;
}

enum Poll<T> {
    Ready(T),
    Pending,
}
```
* `pool`: 通过调用 `poll` 可以使 `future` 前进。如果在本次调用中，`future`完成了，就会返回 `Pool::Ready(result)`。 如果没有完成，就返回 `Pool::Pending`，同时安排一下，当`Future`准备好向下执行时调用 `wake` 。当 `wake` 被调用后，驱动 该 `Future` 的 `executor` 就会再次调用 `poll`。
* 因为 `async fn` 里面也会有 `async fn`, 如果里面 `.await` 了，那么一次的 `poll` 执行也会在此结束，这时会返回 `Pool::Pending` 并安排上 `wake`

举例：如何写一个 `SocketReadFuture` 呢，基本功能为，如果有数据读的是时候，就读取，然后返回结果。如果没有数据读取的时候，就安排 `wake`，并返回 `Pool::Pending`
```rust
pub struct SocketRead<'a> {
    socket: &'a Socket,
}

impl SimpleFuture for SocketRead<'_> {
    type Output = Vec<u8>;

    fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
        if self.socket.has_data_to_read() {
            // The socket has data-- read it into a buffer and return it.
            Poll::Ready(self.socket.read_buf())
        } else {
            // The socket does not yet have data.
            //
            // Arrange for `wake` to be called once data is available.
            // When data becomes available, `wake` will be called, and the
            // user of this `Future` will know to call `poll` again and
            // receive data.
            self.socket.set_readable_callback(wake);
            Poll::Pending
        }
    }
}
```
关于`Future`的调度器，可以按照下面方式实现
```rust
/// A SimpleFuture that runs two other futures to completion concurrently.
///
/// Concurrency is achieved via the fact that calls to `poll` each future
/// may be interleaved, allowing each future to advance itself at its own pace.
pub struct Join<FutureA, FutureB> {
    // Each field may contain a future that should be run to completion.
    // If the future has already completed, the field is set to `None`.
    // This prevents us from polling a future after it has completed, which
    // would violate the contract of the `Future` trait.
    a: Option<FutureA>,
    b: Option<FutureB>,
}

// Join实现了 Future, 所以，Join也是一个 Future
impl<FutureA, FutureB> SimpleFuture for Join<FutureA, FutureB>
where
    FutureA: SimpleFuture<Output = ()>,
    FutureB: SimpleFuture<Output = ()>,
{
    type Output = ();
    fn poll(&mut self, wake: fn()) -> Poll<Self::Output> {
        // Attempt to complete future `a`.
        if let Some(a) = &mut self.a {
            if let Poll::Ready(()) = a.poll(wake) {
                self.a.take();
            }
        }

        // Attempt to complete future `b`.
        if let Some(b) = &mut self.b {
            if let Poll::Ready(()) = b.poll(wake) {
                self.b.take();
            }
        }

        if self.a.is_none() && self.b.is_none() {
            // Both futures have completed-- we can return successfully
            Poll::Ready(())
        } else {
            // One or both futures returned `Poll::Pending` and still have
            // work to do. They will call `wake()` when progress can be made.
            Poll::Pending
        }
    }
}
```
上面只是介绍了一个简单的 `Future`，一个真实的 `Future` 如下所示
```rust
trait Future {
    type Output;
    fn poll(
        // Note the change from `&mut self` to `Pin<&mut Self>`:
        self: Pin<&mut Self>,
        // and the change from `wake: fn()` to `cx: &mut Context<'_>`:
        cx: &mut Context<'_>,
    ) -> Poll<Self::Output>;
}
```

# 使用Waker唤醒task
调用poll的时，future并没有完成的情况非常常见。当碰到这种情况的时候，future 需要保证，“当该future可以往前走的时候，需要被调度器再次polled”。这个保证是由 `Waker` 负责的。

Each time a future is polled, it is polled as part of a "task". Tasks are the top-level futures that have been submitted to an executor.

`Waker` 提供一个 `wake` 方法，这个方法被用来 通知 executor，相关的 task可以被唤醒，向下执行了。当 `wake()` 被调用，executor就知道相关的task可以向下执行了，`Waker` 同样实现了 `clone`，所以它可以被 `copy`

# async/.await
两种方式使用 `async`: `async fn` 和 `async` blocks. 他俩都是返回一个 实现了 `Future trait` 的值
```rust
// `foo()` returns a type that implements `Future<Output = u8>`.
// `foo().await` will result in a value of type `u8`.
async fn foo() -> u8 { 5 }

fn bar() -> impl Future<Output = u8> {
    // This `async` block results in a type that implements
    // `Future<Output = u8>`.
    async {
        let x: u8 = foo().await;
        x + 5
    }
}
```
* `async` bodies 和其它 `futures` 是 lazy 的。they do nothing until they are run。常用的执行 `Future` 的方法是 `.await`
* 当对一个 `Future` 调用 `.await` 的时候，他会尝试将其执行完成。如果执行的过程中 `Future` blocked，它会将控制权交给当前的线程（it will yield control of the current thread）。如果该`Future`又可以继续执行了，`Executor` 就可以拿起该 `Future` ，继续搞。

### `async` 的生命周期
`async` 的生命周期，即：`async` 返回的 `Future` 的生命周期，`async` 只有在 `Future` 被调用的时候，才会执行。对于形参是引用的`async fn`，需要在 引用失效之前调用 `.await`
Unlike traditional functions, async fns which take references or other non-'static arguments return a Future which is bounded by the lifetime of the arguments:

```rust
// This function:
async fn foo(x: &u8) -> u8 { *x }

// Is equivalent to this function:
fn foo_expanded<'a>(x: &'a u8) -> impl Future<Output = u8> + 'a {
    async move { *x }
}
```

One common workaround for turning an async fn with references-as-arguments into a 'static future is to bundle the arguments with the call to the async fn inside an async block。 By moving the argument into the async block, we extend its lifetime to match that of the Future returned from the call to good.
```rust
fn bad() -> impl Future<Output = u8> {
    let x = 5;
    borrow_x(&x) // ERROR: `x` does not live long enough
}

fn good() -> impl Future<Output = u8> {
    async {
        let x = 5; //所以这个x相当于傍上了大腿？
        borrow_x(&x).await
    }
}
```

### `async move`
`async blocks` 和 `closures` 允许 `move` 关键字。`async move` 块的行为类似与传统 `closures`，
* 获取其使用变量的 ownership （后面的都是获取了ownership的副作用）
* allowing it to outlive the current scope
* but giving up the ability to share those variables with other code
```rust
/// `async` block:
///
/// Multiple different `async` blocks can access the same local variable
/// so long as they're executed within the variable's scope
async fn blocks() {
    let my_string = "foo".to_string();

    let future_one = async {
        // ...
        println!("{}", my_string);
    };

    let future_two = async {
        // ...
        println!("{}", my_string);
    };

    // Run both futures to completion, printing "foo" twice:
    let ((), ()) = futures::join!(future_one, future_two);
}

/// `async move` block:
///
/// Only one `async move` block can access the same captured variable, since
/// captures are moved into the `Future` generated by the `async move` block.
/// However, this allows the `Future` to outlive the original scope of the
/// variable:
fn move_block() -> impl Future<Output = ()> {
    let my_string = "foo".to_string();
    async move {
        // ...
        println!("{}", my_string);
    }
}
```


### 多线程executor上使用 .await

当使用多线程 `Future` executor的时候，`Future`可能会在线程之间进行移动，任何的`.await`都有可能导致一次线程的切换。所以使用在 `async` bodies中的变量必须可以在 线程之间移动。

> Future在线程之间的移动表示的含义是，？原始 Future就在那里，不同的线程使用，就需要进行 `clone`，这个`clone` 过程需要Future对象里面的多有元素都是 `Sync` 的。

This means that it is not safe to use `Rc`, `&RefCell` or any other types that don't implement the `Send` trait, including references to types that don't implement the `Sync` trait.

(Caveat: it is possible to use these types so long as they aren't in scope during a call to .await.)

Similarly, it isn't a good idea to hold a traditional non-futures-aware lock across an .await, as it can cause the threadpool to lock up: one task could take out a lock, `.await` and yield to the executor, allowing another task to attempt to take the lock and cause a deadlock. To avoid this, use the Mutex in `futures::lock` rather than the one from `std::sync`.



# Executing Multiple Futures at a Time

* `.await`: 在 `async` bodies 中执行 `future`. 会阻塞当前的task，直到对应的 `future` 完成。
* `join!`:  等待所有的 `future` 完成
* `select!`: waits for one of several futures to complete 

# 总结

rust 异步编程核心： `trait Future `

* `async fn` 关键字：调用异步函数返回一个 `Future`
* `.await` : 用来执行 `Future`
* `Executor`: 调度 `Future`





# 参考资料

[https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html](https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html)





# async & .awati

* `async block` : 实际是一个 `Future object`

* `async fn`: 调用返回的是一个 `Future object`

`Future object` 包含了需要进行的操作，异步的调度器实际就是在调度 `Future object`。



`async block` 代码转成 `future` 的规则是如何的呢？见下面

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

struct Delay {
    when: Instant,
}

impl Future for Delay {
    type Output = &'static str;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<&'static str>
    {
        if Instant::now() >= self.when {
            println!("Hello world");
            Poll::Ready("done")
        } else {
            // Ignore this line for now.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[tokio::main]
async fn main() {
    let when = Instant::now() + Duration::from_millis(10);
    let future = Delay { when };

    let out = future.await;
    assert_eq!(out, "done");
}
```

> async fn main() 会变成什么样的 future 呢？

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

enum MainFuture {
    // Initialized, never polled
    State0,
    // Waiting on `Delay`, i.e. the `future.await` line.
  	// 如果async fn main 中出现了多个 future.await, 这里会出现更多的 状态，State2..., State3...
    State1(Delay),
    // The future has completed.
    Terminated,
}

impl Future for MainFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>)
        -> Poll<()>
    {
        use MainFuture::*;

        loop {
            match *self {
                State0 => {
                  	// 这里实际是 async fn main 的前两行代码
                    let when = Instant::now() + Duration::from_millis(10);
                    let future = Delay { when };
                    *self = State1(future);  // 这里原来是 .await 调用，现在变成了 MainFuture状态的转换。
                }
              	// 在 State1 的时候，就需要等待 .await 的结果出来了。
                State1(ref mut my_future) => {
                    match Pin::new(my_future).poll(cx) {
                        Poll::Ready(out) => {
                          	// 这里是 async fn main 的最后一行代码。在 .await 得到结果之后执行。
                            assert_eq!(out, "done");
                            *self = Terminated;
                            return Poll::Ready(());
                        }
                        Poll::Pending => {
                            return Poll::Pending;
                        }
                    }
                }
                Terminated => {
                    panic!("future polled after completion")
                }
            }
        }
    }
}
```



* `.await` :
  * 在生成匿名`Enum` 时 切分了 `Future` 的 `State`。
  * 在 匿名 `Enum` 中的 `poll` 实现中也对应了一个 `poll` 调用。



# tokio

> Task, executor 调度的基本单位。
>
> 如何实现 task 在 网络请求的时候就挂起。
>
> 如何实现当 task 就绪时，重回调度队列。



三个重点抽象：

* `Executor`: `tokio` 调度器
* `Task`： `tikio` 的基本调度单位
* `Future`: `rust` 提供



想要实现的基本功能：

* `Executor` 中存在一个调度队列，在里面的 `task` 会被 `executor` 轮训调度。

* 调度的 `Task` 如果发生阻塞，`Executor` 就找下一个 `Task` 进行调度。
* 如果阻塞的 `Task` `ready` 了，就回到 `Executor` 的调度队列中。

> 在 mini_tokio 部分会详细介绍，该如何实现



> * `Executor` 执行 `task`
> * `task` 是 `tokio.spawn` 的`async block`
> * `async fn` 里面才能执行 `.await`



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
tokio 的task 是一个 异步`green thread`. 通过 将一个 `async block` 传给 `tokio.spawn` 来创建。`tokio.spawn` 返回一个 `JoinHandle`，调用者可以通过这个 `JoinHandle` 来和 `spawned task` 进行交互。`async block` 可以返回值，调用者可以通过在 `JoinHandle` 上调用 `await` 来获取返回值。

`tasks` 是 `tokio scheduler` 管理的最小执行单元。` Spawning the task` 会将 `task` 提交给 `Tokio scheduler`, 由`Tokio scheduler`决定该如何调度。`spawned task` 可能在 `spawn` 它的线程上执行，也可能在不同的线程上执行。`spawned task` 可以在 不同的线程之间来回移动。 

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

`JoinHandle.await`返回的是`Result`，当`task` 在执行的时候碰到错误(task panic or task is forcefully cancelled by the runtime shutting down)，`JoinHandle.await`就会返回`Err`. 



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

* `Executor` 调度 `Task`
* `Task` 封装了 `future & sender`
  * `future` 代表要执行的操作
  * `sender` 负责当` wake` 的时候，将自己再` send` 给 `executor` 的调度队列
    * 该逻辑应该 `impl ArcWake for Task {fn wake_by_ref()}` 中实现。
  * task 即是 要执行的任务，也是 waker！自己唤醒自己。



```rust
struct MiniTokio {
    scheduled: channel::Receiver<Arc<Task>>,
    sender: channel::Sender<Arc<Task>>,
}

struct Task {
    // The `Mutex` is to make `Task` implement `Sync`. Only
    // one thread accesses `future` at any given time. The
    // `Mutex` is not required for correctness. Real Tokio
    // does not use a mutex here, but real Tokio has
    // more lines of code than can fit in a single tutorial
    // page.
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
    executor: channel::Sender<Arc<Task>>,
}

impl MiniTokio {
    fn run(&self) {
        while let Ok(task) = self.scheduled.recv() {
            task.poll();
        }
    }

    /// Initialize a new mini-tokio instance.
    fn new() -> MiniTokio {
        let (sender, scheduled) = channel::unbounded();
        MiniTokio { scheduled, sender }
    }

    /// Spawn a future onto the mini-tokio instance.
    /// The given future is wrapped with the `Task` harness and pushed into the
    /// `scheduled` queue. The future will be executed when `run` is called.
    fn spawn<F>(&self, future: F)
        where
            F: Future<Output = ()> + Send + 'static,
    {
        Task::spawn(future, &self.sender);
    }
}

impl Task {
    fn schedule(self: &Arc<Self>) {
        self.executor.send(self.clone());
    }
}

impl ArcWake for Task {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        arc_self.schedule();
    }
}
impl Task {
    fn poll(self: Arc<Self>) {
        // Create a waker from the `Task` instance. This
        // uses the `ArcWake` impl from above.
        let waker = task::waker(self.clone());
        let mut cx = Context::from_waker(&waker);

        // No other thread ever tries to lock the future
        let mut future = self.future.try_lock().unwrap();

        // Poll the future
        let _ = future.as_mut().poll(&mut cx);
    }

    // Spawns a new taks with the given future.
    //
    // Initializes a new Task harness containing the given future and pushes it
    // onto `sender`. The receiver half of the channel will get the task and
    // execute it.
    fn spawn<F>(future: F, sender: &channel::Sender<Arc<Task>>)
        where
            F: Future<Output = ()> + Send + 'static,
    {
        let task = Arc::new(Task {
            future: Mutex::new(Box::pin(future)),
            executor: sender.clone(),
        });

        let _ = sender.send(task);
    }

}
```





# Pining

https://rust-lang.github.io/async-book/04_pinning/01_chapter.html

https://docs.rs/pin-project/1.0.8/pin_project/

https://doc.rust-lang.org/stable/std/pin/struct.Pin.html





* Pin 是为了解决 self reference 存在的

* Self reference 问题指的是啥

* 为什么 pin 能够解决 self reference 问题

* pin的底层原理是什么？即：是如何做到 pin 的



> self-referential type: 自引用类型。类型中 **某个字段** 引用了 **类型中的某个字段**

为什么 self-referential type 是个问题呢？仔细看以下代码。

```rust
#[derive(Debug)]
struct Test {
    a: String,
    b: *const String,
}

impl Test {
    fn new(txt: &str) -> Self {
        Test {
            a: String::from(txt),
            b: std::ptr::null(),
        }
    }

    fn init(&mut self) {
        let self_ref: *const String = &self.a;
        self.b = self_ref;
    }

    fn a(&self) -> &str {
        &self.a
    }

    fn b(&self) -> &String {
        assert!(!self.b.is_null(), "Test::b called without Test::init being called first");
        unsafe { &*(self.b) }
    }
}

fn main() {
    let mut test1 = Test::new("test1");
    test1.init();
    let mut test2 = Test::new("test2");
    test2.init();

    println!("a: {}, b: {}", test1.a(), test1.b());
    std::mem::swap(&mut test1, &mut test2);
    println!("a: {}, b: {}", test2.a(), test2.b());

}
```

```
a: test2, b: test1
a: test1, b: test1
```

从上面可以看出，一旦 `self-referential type` 被移动，该对象就不安全了。



那么 pin 对于 `self-referential type` 移动导致的不安全的情况的解决方案是是什么呢？既然移动不安全，那就干脆别移动了。通过rust类型系统来禁止 该对象的移动。并能够在编译期检查出来代码中移动 该对象的情况，并报错？

```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

#[derive(Debug)]
struct Test {
    a: String,
    b: *const String,
    _marker: PhantomPinned, // 这个字段划重点，正式他阻止了 对象的移动
}


impl Test {
    fn new(txt: &str) -> Self {
        Test {
            a: String::from(txt),
            b: std::ptr::null(),
            _marker: PhantomPinned, // This makes our type `!Unpin`
        }
    }
    fn init(self: Pin<&mut Self>) {
        let self_ptr: *const String = &self.a;
        let this = unsafe { self.get_unchecked_mut() };
        this.b = self_ptr;
    }

    fn a(self: Pin<&Self>) -> &str {
        &self.get_ref().a
    }

    fn b(self: Pin<&Self>) -> &String {
        assert!(!self.b.is_null(), "Test::b called without Test::init being called first");
        unsafe { &*(self.b) }
    }
}

pub fn main() {
    // test1 is safe to move before we initialize it
    let mut test1 = Test::new("test1");
    // Notice how we shadow `test1` to prevent it from being accessed again
    let mut test1 = unsafe { Pin::new_unchecked(&mut test1) };
    Test::init(test1.as_mut());

    let mut test2 = Test::new("test2");
    let mut test2 = unsafe { Pin::new_unchecked(&mut test2) };
    Test::init(test2.as_mut());
    println!("a: {}, b: {}", Test::a(test1.as_ref()), Test::b(test1.as_ref()));
    println!("a: {}, b: {}", Test::a(test2.as_ref()), Test::b(test2.as_ref()));
		// 这里的 swap 就会报错了
    std::mem::swap(test1.get_mut(), test2.get_mut());
}
```



```rust
use std::pin::Pin;
use std::marker::PhantomPinned;

#[derive(Debug)]
struct Test {
    a: String,
    b: *const String,
    _marker: PhantomPinned,
}

impl Test {
    fn new(txt: &str) -> Pin<Box<Self>> {
        let t = Test {
            a: String::from(txt),
            b: std::ptr::null(),
            _marker: PhantomPinned,
        };
        let mut boxed = Box::pin(t);
        let self_ptr: *const String = &boxed.as_ref().a;
        unsafe { boxed.as_mut().get_unchecked_mut().b = self_ptr };

        boxed
    }

    fn a<'a>(self: Pin<&'a Self>) -> &'a str {
        &self.get_ref().a
    }

    fn b<'a>(self: Pin<&'a Self>) -> &'a String {
        unsafe { &*(self.b) }
    }
}

pub fn main() {
    let mut test1 = Test::new("test1");
    let mut test2 = Test::new("test2");

    println!("a: {}, b: {}",test1.as_ref().a(), test1.as_ref().b());
    println!("a: {}, b: {}",test2.as_ref().a(), test2.as_ref().b());
}
```





# 其它

一些核心功能的接口定义好。接口定义了就确定了输出是什么类型。这些类型又会对其属性进行一些限制。就达到了安全的目的

