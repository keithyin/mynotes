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

在Rust中`async fn` 创建了一个 `asynchronous function`, 该函数返回一个 `Future`. 执行
