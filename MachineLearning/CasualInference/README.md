# Review
刚开始看uplift modeling时候让我最迷的一点就是对于Treatment Group & Control Group含义的理解。因为之前接触过推荐，所以自然的就理解成 基础策略为Control Group，新策略为Treatment Group。但是实际上并不是这样。拿发红包来说，有一个基础的发券策略S1，A发5元，B发10元。。。 经过模型优化，我们又来了一个策略S2，A发6元，B发5元。S1 和 S2并不是ControlGroup 和 TreatmentGroup。ControlGroup为A发5元，B发5元，C发5元。。。，TreatmentGroup为A发10元，B发10元，C发10元。。。。这个要搞清楚。


## uplift modeling
> the set of **techniques** used to **model the incremental impact** of an action or treatment on a customer outcome。ie：建模 策略 对用户 **影响增益**的一系列工具

* Estimating customer uplift is a causal inference problem: 原因：我们需要评估 策略 作用和不作用在 该人身上的两种结果。当然，一个策略是无法同时 作用和不作用在同一个人身上的，所以 uplift modeling 通常依赖于随机试验
* Estimating customer uplift is a  machine learning problem: 原因：需要训练出一个模型可以进行可靠的 uplift prediction

uplift modeling：
* 训练：根据随机实验收集到的数据训练一个 uplift 模型.
* 预测：根据用户的uplift进行决策




## uplift modeling 三大方法

* Two-Model approach
* Class Transformation approach
* modeling uplift directly


## Causal Inference

符号含义
* $Y_i(1)$ : person i 接受 treatement 后的 outcome  (实验组)
* $Y_i(0)$ : person i 接受 control treatement 后的 outcome (对照组)
* $\tau_i$ : person i 的 causal effect

$$
\tau_i = Y_i(1) - Y_i(0)
$$

我们需要关注的数值为
$$
CATE: \tau(X_i) = E[Y_i(1)|X_i] - E[Y_i(0)|X_i]
$$
其中：$X_i$ 为用户特征。 这个式子也称为 coniditional average treatment effect (CATE)。由于我们无法同时观测到 $Y_i(1), Y_i(0)$, 所以对于 person i，其真实 observed outcome 为 
$$
Y_i^{obs} = W_iY_i(1) + (1-W_i)Y_i(0)
$$
其中： $W_i \in {0, 1}$ 用来表示 person i 是否 接受了 treatment

一个常见的错误是，我们通常使用以下公式来计算 CATE
$$
E[Y_i^{obs}|X_i=x, W_i=1] - E[Y_i^{obs}|X_i=x, W_i=0]
$$
除非我们假设 已知 $X_i$ 条件下 $W_i$ 与 $Y_i^{obs}$ 独立，该公式才和 CATE 计算公式一致。当 treatment assignment is random conditional on $X_i$时，该假设成立。这里需要注意的是在 $X_i$ 条件下进行 random assignment。百度爱迪生那种分流方式也可以认为假设成立。


* propensity score: $P(X_i) = P(W_i=1|X_i)$, the probability of treatment given $X_i$.

# uplift modeling
回忆一下AB测试的流程：随机分流，分为对照组和实验组，在实验组上使用策略，这时候可以认为 “treatment assignment is random conditional on $X_i$”是成立的。然后会看 实验和对照上的一些指标。如果指标是正的，该策略就可以推全了。

那么，uplift modeling是干嘛的呢？是用来建模策略的增量的。ie：预估 $\tau(X_i)$, 预估这玩意有啥用呢？？

如果在AB测试的时候我们知道 $\tau(X_i)$ ，那么我们完全可以使用监督学习的方式学习一个模型进行预测。可惜的是，我们并不知道。uplift领域中有三大主要方法用来预估 $\tau(X_i)$

* Two-Model approach：
  * 构建两个预测模型，一个使用实验组数据，一个使用对照组数据。
* Class Variable Transformation: 用于二值收益变量
  * aa
* model uplift directly through the modification of well known classification machine learning algorithms such as decision tree, random forest or SVM

## Two Model
> 通常被用作 baseline 模型（真的惨。。）
使用 实验组 和 对照组数据 对 $E[Y_i(1)|X_i], E[Y_i(0)|X_i]$ 独立建模。然后使用减法得到 uplift $uplift = E[Y_i(1)|X_i] - E[Y_i(0)|X_i]$
* 独立建模的意思是：两个模型，使用 TreatmentGroup 数据 和 ControlGroup 数据独立训练

广义two models模型：一旦输入特征加上了 control/treatment group的标记，那就是 two models模型了。

## Class Transformation
> 用于 二值收益变量，比如：点击，转化。都是二值收益变量
该方法构建了以下变量
$$
Z_i = Y_i^{obs}W_i + (1-Y_i^{obs})(1-W_i)
$$

* $W_i$ 表示用户 i 是否被 treatment
* $Y_i^{obs}$ 表示 i 是否 点击/转化

$Z_i=1$包含两种情况：
* obs 属于 实验组，且 $Y_i^{obs}=1$
* obs 属于 对照组，且 $Y_i^{obs}=0$

当 $P(X_i=x) = 1/2$ 时，我们可以得到以下公式：
$$
\tau(X_i) = 2P(Z_i=1|X_i) - 1
$$
所以我们只需要建模 $P(Z_i=1|X_i)$，即：$E[Z_i=1|X_i]$. class transformation 方法是优于 two model方法的。所以目前来说比较火🔥

当 $P(X_i=x)=1/2$ 并不满足时：。。。

## Modeling Uplift Directly
> modifying existing machine leaning algorithms to directly infer a treatment effect

# evaluation

* predict uplift for both treated and control observations and compute the average prediction per decile in both groups. Then, the difference between those averages is tenken for each decile. 

> 我们无法看到 treated/not treated在同一个人身上的影响，所以我们没有ground truth来评估uplift model。

* uplift通常使用一些 `aggregated measures` ，例如 `uplift bins` 或者 `uplift curves`. 
* upliff-model的评估通常基于一种假设：相似 uplift-score 的人通常具有相同的行为（认为相同uplift-score的用户是同一类人。）



**Area Under Uplift Curves (AUUC)**

* 准备两个验证集：一个 treatment group验证集，一个 control group验证集。Userid 为 Primary-Key
* 拿训练好的模型预测两组用户 uplift-socres：1）ControlGroup上的用户：计算uplift，2）TreatmentGroup上的用户：计算uplift。
* 两组用户根据uplift-scores进行降序排。该操作用于人群对齐。基于假设（相似uplift-scores的人具有相似的行为）
* 然后分别取两组的top 10%, 20%, ... 100%。计算两组的 转化率差异（并非直接用这个值，而是有一个诡异的公式。）。画出一个曲线
* 曲线下面积即为auuc。
* 离线优化目标可以朝着auuc变大的方向去。即：预测 uplift 高的人群，真实的 uplift 也是高的

uplift高的那些用户，实际上我们计划圈出来进行treatment的，我们需要确认的是，uplift高的人在 treatment 和 control Group 中的表现的确是区分度比较大的，所以才有AUUC？



# Glossary

* uplift modeling:  the set of **techniques** used to **model the incremental impact** of an action or treatment on a customer outcome。ie：建模 策略 对用户 **影响增益**的一系列工具
* quantile: 四分位，概率分布曲线下面积四等分，会得到的三个数
* decile：10分位，
* Percentile


* Customer acquisition： 客户获得（拉新）which prospects are most likely to become customers; this also includes win-back campaigns where attrited customers are targeted;
* Customer development: 客户发展（增加客户在平台上的消费）。which customers are most likely to buy additional products (cross-selling) or to increase monetary values (up-selling);
* Customer retention: 客户维系。which customers are most likely to be ‘saved’ by a retention campaign; this essentially identifies who have more ‘controllable’ risks as opposed to those who will attrite regardless of the retention effort.
* Customer churn: 客户流失
* upselling，cross-selling，down-sell：[https://www.jianshu.com/p/2b7c8ca37c6e](https://www.jianshu.com/p/2b7c8ca37c6e)

# 参考资料
[http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)

[https://www2.deloitte.com/tw/tc/pages/technology/articles/newsletter-12-32.html](https://www2.deloitte.com/tw/tc/pages/technology/articles/newsletter-12-32.html)

[https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247498630&idx=1&sn=b36515e54c2dbc20186942102497c390&chksm=fbd749eacca0c0fc9e285ffc7d06e336115f387394362a4707c71377f02832f8c42bcc71cc7a&mpshare=1&scene=24&srcid=&sharer_sharetime=1585109170232&sharer_shareid=255a68ecb152bdfa3b164d51ce560a8d#rd](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247498630&idx=1&sn=b36515e54c2dbc20186942102497c390&chksm=fbd749eacca0c0fc9e285ffc7d06e336115f387394362a4707c71377f02832f8c42bcc71cc7a&mpshare=1&scene=24&srcid=&sharer_sharetime=1585109170232&sharer_shareid=255a68ecb152bdfa3b164d51ce560a8d#rd)
