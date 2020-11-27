* 什么是 module
  * A module is a collection of Go packages stored in a file tree with a `go.mod` file at its root。比如说我们自己写的项目，如果项目 root目录下有个 `go.mod` 文件的话，那就是 module了。

* module 是用来解决什么问题的
  * module主要是为了解决项目的版本依赖问题，比如说 我们开发机上有多个项目，多个项目都使用了一个相同的 module，但是版本不同，这咋办呢？我们可以手动下载所有的版本，并且给予不同的命名，这样 import 的时候就可以区分开。但是这个方法是非常不灵活的。

* module 如何解决版本依赖问题
  * go module 使用 `go.mod` 文件来管理版本依赖，`go.mod` 中会存储该项目(module) 依赖的 其他 module，并且加上版本信息。
```
module example.com/hello # module’s module path

go 1.12 # go 的版本号

require rsc.io/quote v1.5.2 # 该项目 依赖的 其他 module 及其 版本号
```

# 如果使用 module

* Set Up
```shell
# 开启 module 支持
export GO111MODULE=on

# 设置国内代理
export GOPROXY=https://goproxy.cn
```

* 创建一个 module：cd 到项目的根目录，然后 `go mod init`。当我们执行 `go mod init` 就会看到项目根目录多了一个 `go.mod` 文件，打开进去看

```
module yinpeng/BookMeetingRoom # module's module name. 也是 import 路径

go 1.15 # golang 的版本号
```

* 这时候如果我们要用到别人的包。我们可以在 **项目根目录下执行**  `go get github.com/sirupsen/logrus` (这种方式下载的最新版本)。如果想要使用指定版本，`go get github.com/sirupsen/logrus@v1.6.0`。仔细找找，我们可以看到，下载的 module 存放在了 `$GOPATH/pkg/mod/github.com/sirupsen/logrus@v1.7.0`

* 下载完之后，我们在代码中直接 `import github.com/sirupsen/logrus` 就可以使用了。

* 看到这可能会想，如果拿到了别人代码，里面用了一堆 module，难道我们还得一个个 `go get` 吗？其实大可不必，这里分为两种情况：
  * 代码中 `imported moudles(packages)` 没有写在 `go.mod` 文件中。这时候我们使用 `go run ,go test` 时候，会自动检查 `imported moudles(packages)` 是否都在 `go.mod` 中，如果没有，golang 就会自动下载各个 `modules(packages)` 的最新版本。
  * `go.mod` 中有：这时候我们只需要执行一个命令 让其下载即可 ``


# 命令总结

* 添加依赖
```shell
go get someModule # 在 go.mod同级目录下执行
go run, go test, go build # 会自动将 代码中 imported module添加到 go.mod 中（并下载缺少的 module）
# 也可以通过手动修改 go.mod 文件的方式添加依赖
```
* 管理依赖
```shell
go mod verify # 校验 go.mod 中的依赖 是否有效 （如果我们手动修改 go.mod 版本号，出现没有的版本号的时候，就可以检查出来）
go mod download # 下载 go.mod 中的所有依赖。（避免了我们 一个个的 go get）

go mod tidy # 清理 unused 依赖。
```
