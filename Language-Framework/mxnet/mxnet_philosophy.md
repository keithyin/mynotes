# mxnet philosophy

Symbolic programming, on the other hand, allows **functions to be defined abstractly through computation graphs** . In the symbolic style,

*  we first  **express complex functions in terms of placeholder values** . 
*  Then, we can execute these functions by **binding them** to real values.

> 注释：模型变量，输入，输出，都是 placeholder

> mxnet : 符号编程 自动求导， 命令式编程用来更新参数
>
> mxnet：模型  与 参数 分离



> bind 这个方法就是用来传 参数的
>



## Executor

这个对象中保存了

* 模型参数（包括 模型参数 和 输入）
* 反向传导的梯度



**forward**

执行 forward 操作的时候，先将 输入赋好值，就 ok 了