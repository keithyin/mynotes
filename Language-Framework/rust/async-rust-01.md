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

# 参考资料
[https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html](https://rust-lang.github.io/async-book/01_getting_started/03_state_of_async_rust.html)
