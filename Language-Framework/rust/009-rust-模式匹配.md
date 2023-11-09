# 模式匹配

什么是 `pattern`:

```rust
let x = 5; // x 为 pattern
let Some(x) = SomeOptionValue; // Some(x) 为 pattern
```



`pattern` 有两类

* `refutable`: 可驳倒。
* `irrefutable`: 不可驳倒。

```rust
let x = 5; // x irrefutable. x 是一定可以匹配上的，所以为 irrefutable
let Some(x) = SomeOptionValue; // Some(x) refutable。Some(x) 有可能匹配不上，所以为 refutable
```



 `let, for loops` 仅支持 `irrefutable patterns`. 所以上面两个例子中有个是错误🙅的。

```rust
let Some(x) = SomeOptionValue; // 🙅
if let Some(x) = SomeOptionValue {}; // 🙆
```



`while let, if let` 同时支持 `refutable patterns, irrefutable patterns`

* 但是不建议在 `while let, if let` 中使用 `irrefutable patterns`。在这种地方使用irrefutable patterns就没有条件分支存在的价值了，就是永远走某一个分支了。



`match arms` 必须使用 `refutable patterns` . 但是 `match arms` 的最后一个一定要是一个 `irrefutable pattern`。要保证所有的情况都要关注到。





 
