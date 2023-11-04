# 宏编程

rust 里面两种宏

* 声明宏：做简单的文本替换操作
* 过程宏：根据输入语法树来进行更复杂的操作





编译过程

1. 源代码
2. tokenization (将源代码切成一个个的 `token`)
3. Parsing (`语法解析`)。 一系列 token 构成 `AST(抽象语法树)`
   1. 生成 `AST` 后，宏处理才开始. 这时 宏扩展会作为 `AST` 中的某个节点！
4. 



## Token

rust 中包含多种 token

* 标识符（identifiers, 变量名，函数名，类名，。。。）: `foo`, `Bambous`, `self`, `we_can_dance`, `LaCaravane`. 
* 字面值（literals, 字面值常量）：`42`, `72u32`, `0_______0`, `1.0e-40`, `"ferris was here"`
* 关键字 (keywords): `_`, `fn`, `self`, `match`, `yield`, `macro`
* 符号 （symbols）: `[`, `:`, `::`, `?`, `~`, `@`， （+，-，*，/ ） 是不是也包含在这里



## AST

`a + b + (c + d[0]) + e` 对应的 ast 如下所示

```
              ┌─────────┐
              │ BinOp   │
              │ op: Add │
            ┌╴│ lhs: ◌  │
┌─────────┐ │ │ rhs: ◌  │╶┐ ┌─────────┐
│ Var     │╶┘ └─────────┘ └╴│ BinOp   │
│ name: a │                 │ op: Add │
└─────────┘               ┌╴│ lhs: ◌  │
              ┌─────────┐ │ │ rhs: ◌  │╶┐ ┌─────────┐
              │ Var     │╶┘ └─────────┘ └╴│ BinOp   │
              │ name: b │                 │ op: Add │
              └─────────┘               ┌╴│ lhs: ◌  │
                            ┌─────────┐ │ │ rhs: ◌  │╶┐ ┌─────────┐
                            │ BinOp   │╶┘ └─────────┘ └╴│ Var     │
                            │ op: Add │                 │ name: e │
                          ┌╴│ lhs: ◌  │                 └─────────┘
              ┌─────────┐ │ │ rhs: ◌  │╶┐ ┌─────────┐
              │ Var     │╶┘ └─────────┘ └╴│ Index   │
              │ name: c │               ┌╴│ arr: ◌  │
              └─────────┘   ┌─────────┐ │ │ ind: ◌  │╶┐ ┌─────────┐
                            │ Var     │╶┘ └─────────┘ └╴│ LitInt  │
                            │ name: d │                 │ val: 0  │
                            └─────────┘                 └─────────┘
```





## Token Trees

标记树是介于 标记 (token) 与 AST 之间的东西。

```
a + b + (c + d[0]) + e
```

以上 token stream, 会被解析成下面这样的 token tree

```
«a» «+» «b» «+» «(   )» «+» «e»
          ╭────────┴──────────╮
           «c» «+» «d» «[   ]»
                        ╭─┴─╮
                         «0»
```

> 注意：只有 分组标记 `(...), [...], {...}` 不会被标记作为叶子节点。其余都是叶子节点！







## Rust 中关于宏的语法

下面四种 rust 语法，我们经常会碰到

* `#[$arg]`，使用过程宏的语法: eg，`#[Derive(Clone)]`, `#[no_mangle]`
* `#![$arg]`  ？？？:eg,  `#![allow(dead_code)]`, `#![crate_name="blang"]`
* `$name!$arg` 调用声明宏的语法:  eg,  `println!("Hi!")`, `concat!("a", "b")， `
* `$name!$arg2 $arg1`， 这时定义声明宏的语法:  eg, `macro_rules! dummy { () => {}; }`, 



## 声明宏

声明宏的使用方式为 `$name!$arg` : `$arg` 是怎么传入到声明宏的定义中去的呢？是 `token trees`. `$arg` 必须是一个具有单一 `root` 的，非叶子节点 `token tree`. 意味着 `$arg` 必然是 `(...), [...], {...}` 这种格式。回忆`vec!, println!` 的用法. `vec![1, 2, 3], println!("{}", "hello")` , 其实，`vec!(1, 2, 3) `也是一样的。

定义一个宏, 使用 `macro_rules!`
```rust
#[macro_export]
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```



**宏展开**

在生成 AST *之后* ，对程序进行语义理解之前的某个时间点，编译器将会对所有宏进行展开。

这一过程包括，遍历 AST，定位所有宏调用，并将它们用其展开进行替换。



# 参考资料

https://zjp-cn.github.io/tlborm/macros/syntax/ast.html

https://danielkeep.github.io/tlborm/book/mbe-syn-README.html

https://veykril.github.io/tlborm/print.html
