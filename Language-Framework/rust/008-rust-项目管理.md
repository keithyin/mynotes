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

如何将不同的mod放在不同的文件中。
* rust 中的 crate 树结构需要手动在代码中声明。
* 声明完之后去创建 对应的 文件夹 + 文件即可。
* rust 中：src/lib.rs 或者 src/main.rs 为 crate 的顶层模块。`crate::`，如果如果想要添加新的 `mod` 必须逐级声明下去。

* 假设我们需要分文件放一个 `crate::hello::world` 一个模块。
	* 在 `src/lib.rs` 或者 `src/main.rs` 声明 `hello` 模块, `mod hello`. 并在 创建 `src/hello.rs` 文件。
	* 在 `src/hello.rs` 声明模块 `world` 。`pub mod world` 并创建 `src/hello/world.rs` 文件。
	* 或者 `src/hello.rs` 中写 `mod world{....}`. 这样就不用创建新文件了。

```rust
// src/main.rs
mod hello;

// src/hello.rs
pub mod world

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

