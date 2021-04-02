> ContextualBandits: 
>
> * decision making 问题。
> * 奖励是 partially observed. 因为没有办法一次性给用户所有的可选action让其选择。
> * 奖励可以建模为 $r(state,action)$



# 以广告系统为例

* 广告投放就是一个 context bandits 问题。
* 用户的状态($state$)，这次应该给用户展现什么广告 ($action$)，用户对该广告进行反馈 ($reward$)



在做广告的CTR预估业务时，经常碰到的两个场景是：

1. 原来的模型已经时候最新的数据训练过了。新参数在上线之前需要做什么验证吗？
2. 发现了个新模型结构，赶紧试一试。离线训练得到了个AUC，这时候应该做什么验证才能使该模型结构上线？



对于第二个场景，基本操作流程为：

1. AB测试，如果结果OK就可以上线了。

对于第一个场景，实际上也可以做一下AB测试就OK。



> AB测试，真香。



但是，如果AB测试成本过高，咋整呢？

* 少开测试流量？
* 换种方式？



> Counterfactual Policy Evaluation 就是另一种方式。
>
> 通过历史旧策略的数据，而非AB实验获取新数据的方式，对新策略进行效果评估。



# Counterfactual Policy Evaluation

* 方法一:
  * Direct Method: 从历史数据中学习出 reward function。然后根据学习出的 reward function来评估新策略。
  * 这个方法存在一个问题：当训练集与Inference数据分布不一致的时候，即使训练集可以得到的很好的结果，那么Inference也会一塌糊涂。旧Policy产生的数据就是训练集，新策略产出的 action 就是 Inference数据，效果也不会很准。
* 方法二：
  * Inverse Propensity Score: 使用 importance weighting 纠正历史数据中 action 的分布。
  * 这个方法存在的问题是：方差太大。
* 方法三：
  * Doubly Robust



