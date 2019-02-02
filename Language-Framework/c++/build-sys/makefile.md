# makefile

* 说明了编译文件的步骤
* 最开头的部分是终极目标

**makefile 三要素**

* 目标, 依赖, 命令
  * 目标: 目标文件
  * 依赖: 生成目标文件需要的依赖文件
  * 命令:  使用依赖生成目标文件的 命令

```cmake
目标: 依赖
	命令
app: main.c sub.c mul.c
	gcc main.c sub.c mul.c -o app
```



* 如果不想所有文件都被编译, 编译过程就要分开写了.
  * 第一个是终极目标, **其余都是为生成终极目标服务的**
  * 这么分开写, 会检查当前 `.o` 文件对应的 源文件是否被更新
  * 是否更新的原理:
    * 比较**目标** 和 **依赖** 的最终修改时间

```cmake
app: main.o sub.o mul.o
	gcc main.o sub.o mul.o -o app

mian.o: main.c
	gcc -c main.c

sub.o: sub.c
	gcc -c sub.c

mul.o: mul.c
	gcc -c mul.c
```





```shell
# makefile 的工作原理
# 1. 先看终极目标, 查看依赖
# 2. 如果依赖不存在或者过期, 就去下面找生成依赖的规则
```







* **makefile变量**
  * 模式规则匹配
    * `%.o: %.c`
  * 自动变量: 只能在命令中使用
    * `$<` 规则中的第一个依赖
    * `$^` 规则中的所有依赖
    * `$@` 规则中的目标
  * makefile 自己维护的变量: **大写的, 用户可以自己设置**
    * `CPPFLAGS`: 预处理器需要的选项,如 : `-I`
    * `CFLAGS`: 编译的时候需要的参数, `-Wall -g -c`
    * `LDFLAGS`: 链接库使用的选项 `-L  -l`

```cmake
obj=main.o sub.o mul.o
target=app # 赋值操作, ${} 取值操作

${target}: ${obj}
	gcc ${obj} -o ${target}

# 模式规则匹配, %会进行匹配,
%.o:%.c
	${CC} -c $< -o $@
```



* **函数**
  * 所有 `makefile` 中的函数都是有返回值的
  * `wildcard ./*.c `获取到指定路径下的所有 `.c` 文件
  * `patsubst ./%.o, ./%.c, $(src)` : 替换

```cmake
src=$(wildcard ./*.c)


# make clean: 用来执行这个目标, 因为没有依赖,直接执行这个命令就行了
.PHONY:clean # ???? 这是干嘛的
clean:
	rm *.o
	-rm *.o # - 如果执行失败, 忽略, 继续向下执行
```

