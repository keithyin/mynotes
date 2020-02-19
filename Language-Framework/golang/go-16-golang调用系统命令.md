### `exec.Command`

需要注意的几个点

* 如果执行命令
  * 是同步的还是异步的, 如果是异步的, 应该如何等待.
* 如何获取程序的标准输出
  * `cmd.Stdout = new(bytes.Buffer)`
* 如果获取程序的标准错误输出
  * `cmd.Stderr = new(bytes.Buffer)`
* 如果确认程序的退出状态码



编码流程

* 构建 `cmd` 对象
  * `cmd := exec.Command("ls", "-a", "-l")`
* 准备 `buffer` 接收 标准输出 和 标准错误输出
  * `var stdout bytes.Buffer`
  * `cmd.Stdout = &stdout`
* 执行 命令
  * 同步 `cmd.Run()`
  * 异步`cmd.Start()`, 等待 `err:=cmd.Wait()`, `err  里面有 exit status` 信息, 只不过被包装了 `error`
* 这时候可以打印 输出结果了
  * `fmt.Println(stdout.String())`



```go
package main
 
import (
    "io/ioutil"
    "log"
    "os/exec"
  	"bytes"
)
 
func main() {
    // 执行系统命令
    // 第一个参数是命令名称
    // 后面参数可以有多个，其实就考虑成 我们命令行键入的命令 用 ' ' split 一下.
  	// 此时并未执行
    cmd := exec.Command("ls", "-a", "-l")
    // 获取输出对象，可以从该对象中读取输出结果
    var stdout bytes.Buffer
    
    // 运行命令, err string 里面是有 error status 255  这种玩意的.
    if err := cmd.Run(); err != nil {
        log.Fatal(err)
    }
  	log.Println(stdout.String())
}
```





