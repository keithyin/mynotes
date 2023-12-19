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





# let mut

* `mut` 两个作用
  * 变量所代表的 `栈空间` 的值是否可变。注意⚠️：这里没有提堆的事情哦。
  * 变量是否可以 mutable reference
    * 会影响到一些方法的调用，因为一些方法需要 `&mut self`.  需要修改栈空间值的，才需要 `&mut self`
  * 总结起来看：`mut` 仅表示，变量所表示的栈空间上的值时候能变。
* 有了引用，自然会有解引用
  * `*some_obj` :
    * 当 `some_obj` 是 `T` 的时候，该代码会展开 `*(some_obj.deref(&self)) or *(some_obj.deref_mut(&mut self))` 
    * 当 `some_obj` 是 `&T, &mut T` 时候，就会返回 `T, mut T` 了。

# 几个特殊类型

* `()`: 等价于 void，返回 空 。
* `Never`: `!` Never类型，表示不返回数据。 panic("") 就是返回 Never

* PhantomData: 幽灵数据。不表示任何数据，一个Zero-Type类型，主要是为了将 泛型，生命周期标记 绑定到上面的。

PhantomData用法：
```rust
// 和裸指针一起使用, 因为裸指针是没办法附着声明周期的，可以将声明周期 附着在 PhantomData上
// 为什么要加声明周期标注？ 因为 SomeRef 对象有可能会绑定到 其它生命周期的变量上，如果加了 生命周期标注，就能在编译期进行 生命周期检查，减少错误发生。

struct SomeRef<'a, T> {
    data_ptr: *const T,
    _phantom: PhantomData<&'a T>
}

impl<'a, T: 'a> SomeRef<'a, T> {
    pub fn new(ptr: &'a T) -> SomeRef<T> {
        SomeRef{
            data_ptr: ptr as *const T,
            _phantom: PhantomData
        }
    }

    pub fn get_ref(&self) ->&T {
        unsafe {&*self.data_ptr}
    }
}


```



# 其它资料

https://stackoverflow.com/questions/31567708/difference-in-mutability-between-reference-and-box
