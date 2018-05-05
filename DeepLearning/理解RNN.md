# 理解 RNN

* 针对时序信息建模
* $t$ 时刻的输出由 $t$ 时刻的输入和 $t$ 时刻的隐层状态决定 。



## LSTM

RNN 存在梯度消失的问题，梯度消失指的是 后面时刻的 `loss` 无法传到前面去，所以提出了 `LSTM`

* 增加了一条告诉公路，这条路上没有 非线性激活单元
* 只有 加法 和 乘法操作
* 一定程度上缓解了 梯度消失的问题，**没有根本上解决，因为 乘 0 就凉了**





## 参考资料

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)