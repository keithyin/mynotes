Tmux入门教程
WHY
使用Tmux有很多好处。我最看重的是，

远程服务器上持久地保存工作状态。比如，我在本地或服务器做某项工作，需要在Terminal打开几个窗口。每次换工作，还需要切换。用Tmux，便可以按照Session进行管理，每次轻松打开切换。
随时随地断开或连接Session。比如，在服务器上面运行一条命令，但是担心连接断开后命令终止，此时在tmux中运行命令，此时断网关机便问题不大，只需随后恢复Session便可以。
Install
服务器上需要根据具体操作系统安装，包管理一般支持。若不支持，请参考GitHub - tmux/tmux: tmux source code。 Mac上只需如下命令：

brew install tmux
复制代码
Config
Tmux默认不支持鼠标滚动查看窗口中的前后内容。需要修改配置来支持。

touch ~/.tmux.conf

# 开启鼠标模式
set-option -g mouse on

# 允许鼠标选择窗格
# set -g mouse-select-pane on
​
# 如果喜欢给窗口自定义命名，那么需要关闭窗口的自动命名
set-option -g allow-rename off
​
# 如果对 vim 比较熟悉，可以将 copy mode 的快捷键换成 vi 模式
set-window-option -g mode-keys vi
复制代码
如果对鼠标滚动效果依旧不满意，可以参考这篇blog：Better mouse scrolling in Tmux

Conception
tmux的主要元素分为三层：

Session 一组窗口的集合，通常用来概括同一个任务。session可以有自己的名字便于任务之间的切换。
Window 单个可见窗口。Windows有自己的编号，也可以认为和ITerm2中的Tab类似。
Pane 窗格，被划分成小块的窗口，类似于Vim中 C-w +v 后的效果。 一图以蔽之：
![](https://user-gold-cdn.xitu.io/2018/2/18/161a7829cd0bc4b2?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)
Session
Tmux为了防止与全局快捷键冲突，大部分快捷键需要先需要输入前缀Ctrl + b，下文用Prefix代替。

Session主要相关命令如下：

# 创建
tmux new # 不指定session name
tmux new -s [session-name]

# 删除Session
tmux kill-session -t [session-name]
tmux kill-server

# 列出当前Session
tmux ls # 
Prefix s # tmux 内

# 恢复Session
tmux a -t [session-name]
tmux a

# 断开Session
tmux detach
Prefix d

# 重命名Session
Prefix $
复制代码
Window
# 创建
Prefix c

# 选择窗口
Prefix + [number] # 选择第n个窗口
Prefix + p/n	# 前/后一个窗口

# 关闭窗口
Prefix &
exit

# 列出所有window（包含其他Session）
Prefix w 
j/k # 前后选择

# 搜索窗口
Prefix f

# 重命名当前窗口
Prefix ,
复制代码
Pane
# 创建
Prefix %	# 水平窗格
Prefix '"'	# 垂直窗格

# 关闭
Prefix x

# 切换
Prefix o # 在窗格间切换
Prefix q	# 显示窗格编号，输入编号切换

# 将当前窗格切换到新窗口
Prefix ！

# 窗格交换位置
Prefix + {/}
复制代码
