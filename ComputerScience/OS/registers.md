# 寄存器

## 通用寄存器
> 应用程序代码可以随便用的

| 寄存器 | 描述        |
| ------ | ----------- |
| %rax   | 返回值      |
| %rbx   | callee 保存 |
| %rcx   | 第4个参数   |
| %rdx   | 第3个参数   |
| %rsi   | 第2个参数   |
| %rdi   | 第1个参数   |
| %rbp   | callee保存  |
| %rsp   | 栈指针      |
| %r8    | 第5个参数   |
| %r9    | 第6个参数   |
| %r10   | caller保存  |
| %r11   | caller保存  |
| %r12   | callee保存  |
| %r13   | callee保存  |
| %r14   | callee保存  |
| %r15   | callee保存  |
|        |             |
|        |             |
|        |             |

具有特殊用途的指针

* `%rsp`: 指向栈顶，`push, pop` 操作会对该寄存器又副作用
  * 也可以显示对该寄存器操作。`sub $20, %rsp`

callee保存：**被调用者保证** 该寄存器的值在 `调用返回时` 和 `调用发生时` 的值是一样的。

* `%rbx, %rbp, %r12, %r13, %r14, %r15`
* 即：如果想用这几个寄存器的话，那就记得先将这几个寄存器的值保存起来（栈里）。在函数返回之前复原



caller保存：如果**调用者**使用了这些寄存器，那么在执行 `call` 指令前，要将这些寄存器的值保存。在`call` 之后复位

* `%r10, %r11, %rax, %rcx, %rdx, %rsi, %rdi, %r8, %r9`
* 即：除了 `%rsp` 和 callee 保存的寄存器，剩下的都是 `caller 保存`



管理变长栈帧：

* `%rbp`: base pointer, 基指针 

## GDTR & IDTR
* DGTR: global descriptor table register。指明全局描述表位置的寄存器
* IDTR: interrupt descriptor table register。指明中断描述符表的寄存器

```asm
lgdt operand
lidt operand
```
* operand：是一个6bytes(48bits)的值。前16位表示表的大小，后32位表示表的基址。
https://c9x.me/x86/html/file_module_x86_id_156.html

注意：
* The LGDT and LIDT instructions are used only in operating-system software; they are not used in application programs
* They are the only instructions that directly load a linear address (that is, not a segment-relative address) and a limit in protected mode
* They are commonly executed in real-address mode to allow processor initialization prior to switching to protected mode.


## 段寄存器
* `cs`: 代码段寄存器
* `ds`: 数据段寄存器
* `es`: 附加段寄存器
* `ss`: 栈段寄存器



# 系统调用

系统调用的所有参数都是通过通用寄存器传递的。

* `%rax` : 存放系统调用号
* `%rcx, %rdx, %rsi, %rdi, %r8, %r9`: 用来存放系统调用参数
* `syscall` : 系统调用指令
