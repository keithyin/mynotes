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

# variable


A variable is a component of a stack frame, either a named function parameter, an anonymous temporary, or a named local variable.
* stack frame：分配给函数的栈内存，被用来存储所有的 **局部变量和函数参数**

```rust
// main函数的 stack frame 足够放下 两个int 和 一个 f32.
// 当main返回的时候，分配给 main 的 stack frame 会被释放。
// stack frame 分配 和 释放的 优雅之处在于：1）分配和释放不需要用户来控制。2）分配的大小可以由compiler计算出来，因为编译器知道函数里面用了哪些局部变量。
fn main() 
{ 
    let a = 10; 
    let b = 20; 
    let pi = 3.14f32; 
} 
```

A local variable (or stack-local allocation) holds a value directly, allocated within the stack's memory. The value is a part of the stack frame.

Local variables are immutable unless declared otherwise. For example: let mut x = ....

感觉：`let mut x = ...`: 只是用来表示，stack frame 上的值能不能修改。那么为什么修改 堆上的值 也需要 `let mut x = ...` 呢？ 有些堆上值的修改，并不会影响栈上值的改变呀。虽然有些是会的。
