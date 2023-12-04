rust中，将错误处理分为两类：1)不可恢复错误 `panic!` 。2）可恢复错误 `enum Result`



# panic!

`panic!` 宏：会终止代码的执行



# Result

https://doc.rust-lang.org/std/result/

```rust
enum Result<T, E> {
  Ok(T),
  Err(E),
}
```

* `Result` 定义了两个变体：代码执行成功时，返回 `Ok(T)`, 如果执行失败时，返回`Err(E)`. 我们可以通过判断返回的是啥 来判断代码是否执行成功。

对Result对象的处理

* 失败时触发 `panic`. :
  *  `.unwrap()`: 直接抛异常
  *  `.expect(msg)`：抛异常时候还会带 msg，知道是在什么地方发生的问题
  *  `result.unwrap_or(fallback)`：如果错误了，就返回 fallback
  *  `result.unwrap_or_else(fallback_fn)` 如果计算 fallback值成本高的话，可以用这个
* 错误传播：`? 运算符` 。如果发生了错误就立刻返回！
* 最原始方法：使用 `match`
```rust
if let Ok(v) = SomeResult {
} else {
  // exception process
}
```

* `.ok()` 转成 Option 来处理

# Option
这个不是处理错误的，这个是用来封装有可能是None 的值的。

https://doc.rust-lang.org/std/option/

```rust
enum Option <T> {
  Some(T),
  None,
}
```



 
