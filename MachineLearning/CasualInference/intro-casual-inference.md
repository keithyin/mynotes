符号基本含义

* $Y_i(0)$ : 表示如果对第 $i$ 个人不施加 Treatment，那么这个人的 potential outcome 是多少
* $Y_i(1)$ : 表示如果对第 $i$ 个人**施加** Treatment，那么这个人的 potential outcome 是多少
* $(Y_i(1), Y_i(0)) \perp\perp T$ : 表示，用户的 potential outcome 与其是否接受了 treatment 无关！
  * 这个条件也叫做 exchangeability: 即，将 treatment 组与 non-treatment 组的人群互换，会得到相同的结果
* $(Y_i(1), Y_i(0)) \perp\perp T | X$
  * conditional exchangeability: 给定了 条件 X 后，就满足了 exchangeability
  * Unconfoundedness, conditional ignorability, conditional exchangebility (一堆名字)
* 



# 疑问

$do(t)$ 会修改原来的因果图，那么如何做到 $do(t)$ 呢？使用随机试验可以做到 $do(t)$ , 还有其它方式可以做到吗？感觉 $do(t)$推的公式都很迷惑。如果想 obervational data 搞到 $do(t)$ 的效果该怎么搞呢？
