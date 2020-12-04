# go 的包管理

* GOPATH
* vender
* go module

# GOPATH

* 需要用的库 + 自己开发的代码都要放到 `$GOPATH/src` 目录下，`import` 导入的路径查找也是 `$GOPATH/src`

问题：

* 多版本的时候，不好处理。

# vender

* 工程 root 下有个 vender 子目录，用于放置依赖包，这样不同项目使用不同包的版本就更容易配置了。

问题：

* 没有依赖包 的版本信息

# Go Module

* 增加了版本信息 + 包的分版本下载。

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
go mod init # 如果不加参数的话 生成的 go.mod 第一行就是该项目 src/ 后面根的路径
go mod init github.com/keithyin/ProjName # 加参数的话， 这个参数就构成了 go.mod 的第一行。
```

```
module yinpeng/BookMeetingRoom # module's module name. 也是 import 路径。如果别人使用该项目的话, import 时候要填入该路径！

go 1.15 # golang 的版本号
```

* 这时候如果我们要用到别人的包。我们可以在 **项目根目录下执行**  `go get github.com/sirupsen/logrus` (这种方式下载的最新版本)。如果想要使用指定版本，`go get github.com/sirupsen/logrus@v1.6.0`。仔细找找，我们可以看到，下载的 module 存放在了 `$GOPATH/pkg/mod/github.com/sirupsen/logrus@v1.7.0`

* 下载完之后，我们在代码中直接 `import github.com/sirupsen/logrus` 就可以使用了。

* 看到这可能会想，如果拿到了别人代码，里面用了一堆 module，难道我们还得一个个 `go get` 吗？其实大可不必，这里分为两种情况：
  * 代码中 `imported moudles(packages)` 没有写在 `go.mod` 文件中。这时候我们使用 `go run ,go test` 时候，会自动检查 `imported moudles(packages)` 是否都在 `go.mod` 中，如果没有，golang 就会自动下载各个 `modules(packages)` 的最新版本。
  * `go.mod` 中有：这时候我们只需要执行一个命令 让其下载即可 ``


# 命令总结

golang 对于 module 的命令 一般都包含两个影响：1）修改 go.mod 文件，2）下载对应的 module

| 命令              | go.mod                                      | 下载 | 补充说明                                                     |
| ----------------- | ------------------------------------------- | ---- | ------------------------------------------------------------ |
| go get            | 修改依赖的版本，或者添加一个新依赖          | Yes  | 需要在 go.mod 所在目录下执行？                               |
| go run/test/build | imported module 如果不在 go.mod中，则会添加 | Yes  |                                                              |
| go mod verify     | 验证go.mod中依赖的合法性                    | No   |                                                              |
| go mod download   | 下载 go.mod中的依赖                         | yes  |                                                              |
| go mod tidy       | 添加module依赖 & 删除不用依赖               | Yes  |                                                              |
| go list -m all    |                                             |      | 打印该module所有的依赖，是go.mod中的，还是所有 imported 就不确定了。 |

* `go list -m -versions rsc.io/sampler` 可以查看一个 `module` 都有哪些版本
* 

# 发布 module

1. 为项目开一个 github repo
2. 项目目录下 执行 `go mod init github.com/keithyin/ProjName`
3. 项目开发
4. 打tag： `git tag v1.0.0` , 关于版本号命名规范 https://blog.golang.org/publishing-go-modules
5. push到 github： `git push origin v1.0.0`

# 参考资料
https://blog.golang.org/using-go-modules
https://zhuanlan.zhihu.com/p/311969770
