# GDB 调试

* 第一步: 使用 `-g` 标记编译程序, `gcc main.c -o main -g`
* 第二步: 进入 gdb, `gdb main`



**查看源代码**

```shell
# l filename.c:行数
# l filename.c:函数名
# l           //继续往下看
# 直接回车,啥都不输入,默认执行上一条指令
```



**打断点**

```shell
# break 行数 // 在第几行打印断点
# b     行数  // 同样功能
# b 15 if i==5  // 第15行打一个条件断点
# info b          // 查看断点信息,  information break 的简写
# del 断点对应的编号  // 删除断点
```



**开始调试**

```shell
# start // 开始调试, 只执行一步
# run   // 直接运行到 第一个断点位置
# n      // 单步调试, 无法进入函数体内部, next
# c      // 继续执行, 执行到断点
# s      // 可以进入到函数体内部, step
# p  var_name  // 查看对应变量的值
# ptype var_name  // 查看类型
# display var_name  // 追踪变量的值
# undisplay 变量编号  // 不追踪了
# u                 // 跳出循环
# finish            // 跳出当前函数
# set var=10        // 设置变量的值, 可以控制 for 循环进程


## 如果碰到 fork
# set follow-fork-mode child 跟踪子进程
# set follow-fork-mode parent 跟踪父进程
```

