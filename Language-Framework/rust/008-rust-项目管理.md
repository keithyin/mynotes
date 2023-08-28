# packages and crates

* 包：`cargo` 提供一个功能，允许 构建，测试，分享create
    * `cargo new` 创建 package
    * package 包含一个或多个 crate
    * package 包含 `Cargo.toml` 文件。该文件描述该如何编译这些 crates。
    * 关于package中可以包含什么东西有一些限制：1）至少包含一个 `crate`。2）至多包含一个 `lib crate`. 3）可以包含任意多 `binary crate`
    *
* crate：一个模块的树形结构，形成库或二进制项目。 crate 指模块的树形结构
    * `crate root`: 是一个源代码文件。the Rust compiler starts from and makes up the root module of your crate

**关于 `cargo new`**
当我们执行`cargo new my_project`命令时，`cargo` 会创建一个名为 `my_project`的文件夹，里面包含一个 `Cargo.toml` 文件，还有一个 `src/main.rs` 文件。可以看一下`Cargo.toml`，里面没有提到`src/main.rs`。这是因为`rust`遵循 "src/main.rs是 binary crate的 crate root，crate名字与 package 名字一致"。同样，"src/lib.rs是 library crate的 crate root，crate名字与 package 名字一致"。`cargo` 将 `crate root` 传给 `rustc` 由其编译 `library or binary`.

当一个package仅包含 `src/main.rs`，说明其仅包含一个名为 $package_name `crate`。当一个package同时包含 `src/main.rs, src/lib.rs`，它将有两个`crate`，一个 `binary`, 一个`library`，两个的名字都是 $project_name. 通过将crate放到 `src/bin/`目录下，一个package可以有多个 binary crate，每个文件都是一个 binary crate.


* 创建一个binary  `cargo new some_proj --bin`  
* 创建一个lib     `cargo new some_proj --lib`
* 模块：通过 use 来使用，用来控制作用域和路径的私有性
* 路径：一个命名例如结构体、函数或模块等项的方式。
* 包与create
    * 包提供一系列功能的一个或多个create
    * crate root 是 src/main.rs 或者是 src/lib.rs。如果只有 main.rs，则说明这个包只有一个 crate。如果同时包含 main.rs 和 其他的 lib.rs，则说明有多个 crate
    * crate 会将一个作用域的相关功能 分组到一起，使得该功能可以很方便的在多个项目之间共享
* 使用模块控制作用域和私有性
    * 创建一个 lib 可以通过 `cargo new --lib libname` 来进行创建
    * 默认所有项（函数，方法，结构体，枚举，模块，常量） 都是私有的，需要使用 `pub` 才能暴露给外部

```rust
mod factory {
    mod factory2 {
        fn produce() {
            
        }
    }
}

mod public_mod {
    pub mod public_mode_inner {
        pub produce() {
            
        }
    }
}

fn main() {
    factory::factory2::produce();
}
```

# modules 和 访问控制

* `use keyword`: brings a path into scope
* `pub keyword`: make items public
*  moduels:
    *  let us organize `code within a crate` into groups for readability and easy reuse.
    *  control the `privacy` of items
*  `src/main.rs, src/lib.rs`：这俩称之为 `crate roots` 的原因是: 他俩构建了一个名为 `crate` 的module，该`module` 为 `crate module tree` 的 root。
    * Notice that the entire module tree is rooted under the implicit module named `crate`.

```rust
// src/lib.rs
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}

        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}

        fn serve_order() {}

        fn take_payment() {}
    }
}
```
该文件的 module tree 为
```
crate
 └── front_of_house
     ├── hosting
     │   ├── add_to_waitlist
     │   └── seat_at_table
     └── serving
         ├── take_order
         ├── serve_order
         └── take_payment
```

## 路径 & 访问控制
路径有两种形式
* 绝对路径：从 `crate::` 开始，或者 `crate_name::`
* 相对路径：从当前路径开始，使用 `self, super, mod_name`
* 访问控制：sibling 可以互相访问。可以访问直系祖辈。其它的只有 pub 标记才能访问。


```rust
mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

pub fn eat_at_restaurant() {
    // Absolute path, front_of_house也是 private的，但是由于 sibling，所以可访问。eat_at_restaurant 和 mode front_of_house 是 sibling
    crate::front_of_house::hosting::add_to_waitlist();

    // Relative path
    front_of_house::hosting::add_to_waitlist();
}
```

```rust
fn serve_order() {}

mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order();
    }

    fn cook_order() {}
}
```

## use 关键字
> Bringing Paths into Scope with the use Keyword.
>

we bring the crate::front_of_house::hosting module into the scope of the eat_at_restaurant function so we only have to specify hosting::add_to_waitlist to call the add_to_waitlist function in eat_at_restaurant.
```rust
use std::io::{self, Write}; // self 表示的是 use std::io
use std::collections::*; // * 是将collections 下的所有一级 item 搞到当前 scope 下。

mod front_of_house {
    pub mod hosting {
        pub fn add_to_waitlist() {}
    }
}

// self::front_of_house::hosting; 相对路径
use crate::front_of_house::hosting; // 绝对路径
// use crate::front_of_house::hosting as other_name; // 通过 as 提供另一个名字

// pub use crate::front_of_house::hosting; // re-export

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
    hosting::add_to_waitlist();
}
```

## 将 module 切分成多个文件

如何将不同的mod放在不同的文件中。

* rust 中的 crate 树结构需要手动在代码中声明。即：使用 `mod ...` 来进行 `module tree`声明。
  * `mod` 只能一级一级的声明，不能一次声明多级。比如 `mod a::b` 就是🙅的。
* 声明完之后去创建 对应的 文件夹 + 文件即可。
* `math.rs` 文件中声明的 `mod rnd;` 对应于 `math/rnd.rs` 文件。
  * 有一个特殊情况，那就是 `src/lib.rs` 文件中声明的 `mod some_mod;` 是对应 `src/some_mod.rs` 文件。而非 `src/lib/some_mod.rs` 。只有 `crate root` 文件中的声明特殊而已。 
* rust 中：src/lib.rs 或者 src/main.rs 为 crate 的顶层模块。`crate::`，如果如果想要添加新的 `mod` 必须逐级声明下去。
* 假设我们需要分文件放一个 `crate::hello::world` 一个模块。
  * 在 `src/lib.rs` 或者 `src/main.rs` 声明 `hello` 模块, `mod hello`. 并在 创建 `src/hello.rs` 文件。
  * 在 `src/hello.rs` 声明模块 `world` 。`pub mod world` 并创建 `src/hello/world.rs` 文件。
  * 或者 `src/hello.rs` 中写 `mod world{....}`. 这样就不用创建新文件了。

```rust
// src/main.rs
mod hello; //理解为模块声明。编译的时候rust前往模块同名文件中 加载模块内容。

// src/hello.rs
pub mod world; 

// src/hello/world.rs
pub fn call() {
	println!("hello world");
}
```

```rust
// src/main.rs
mod hello;

// src/hello.rs
pub mod world{
	pub fn call() {
		println!("hello world");
	}
}
```

* 总结，如果想把代码 分 文件(夹) 管理起来，那就首先在代码中声明好，然后对应的文件创建起来，然后将代码移动过去

