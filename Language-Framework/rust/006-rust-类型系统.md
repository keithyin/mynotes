rust 的类型有 `T, &T, &mut T`

* `&mut` 是一个整体！！ 不能分开看！！

# rust 中的 `&, &mut`

* 和类型放在一起：表示 类型 `&T, &mut T`. `&i32, &mut i32`.  类型`T` 的 引用/借用 类型
* 等式右边：表示引用/借用操作。返回一个 引用/借用 类型。
* 等式左边（变量声明部分）：模式匹配

```rust

// &, &mut 和类型一起时候，是用来表示一个类型！
fn some_func(a: &mut i32){

}

fn main(){
    let mut a = String::from("hello world");

    let ref_a = &mut a; // &mut 放在等式右边 表示：借用该变量，会产生一个 &mut T 类型的变量（ref_a）
    ref_a.push_str("hhhh");

    let &mut ori_a = ref_a; // &mut 放在右边，用来进行模式匹配，这时ori_a 就是一个 String 类型。所以该操作等价于 `let ori_a = *ref_a`。会发生move。所以会报错。
    println!("{}", a);  
  
}
```

# rust 中的 `ref, ref mut`

绑定引用，而非绑定原始值
```rust
let a = 5;
let ref ref_a = a; // 等价于 let ref_a = &a;
```
