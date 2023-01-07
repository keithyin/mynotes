# tokio

> 异步的运行时
> 
> 区分顶层Future和非顶层Future，tokio的调度最小粒度是顶层Future

什么时候不用Tokio

1. 想要加速 `Cpu-bound computation` 时不应该用 `tokio` 这时可以用 `rayon`. `Tokio` 是用来解决 `IO-bound computation` 的

2. 读大量文件，`tokio` 对于读大量文件的应用来说和普通的线程池差不多，这是因为操作系统通常不会提供读文件的异步操作

3. 仅发一个web请求。`tokio` 的优势是并发，非并发场景没必要用
