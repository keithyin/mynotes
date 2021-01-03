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



# 参考资料
[https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html](https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html)
