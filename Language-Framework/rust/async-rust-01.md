rust的异步编程模型

* 异步方法：`async fn` ，调用一个 `async fn` 实际返回一个 `Future`. 这时实际上是没有什么计算的。

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
