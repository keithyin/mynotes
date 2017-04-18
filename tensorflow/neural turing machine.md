# Neural Turing Machine
[论文地址](https://arxiv.org/pdf/1410.5401.pdf)
[tensorflow源码地址](https://github.com/carpedm20/NTM-tensorflow)
图
<center>NTM结构图</center>
图中：A为控制器（为LSTM单元）write为write head, read 为 read_head
源码重点注释：
## ops.py
ops.py中有两个函数，分别为linear() 、 Linear() ,这两个函数有什么区别呢， 看上图， linear
主要用于左下角， Linear用于右上角

## ntm_cell.py
build_controller():用于生成LSTM单元，用于NTM的LSTM单元和常见的不同，它由三个部分构成了输入，
build_read_head(): 生成read head， 就是图中的read
