> 强化学习在推荐系统中的应用

* 如果将用户与信息流产品的交互看做 序列决策过程的话 (用户作为 环境, 推荐系统作为 agent).  这么做强化学习需要注意什么?
  * 状态转移概率的建模?

#Deep Reinforcement Learning for Online Advertising in Recommender Systems(2019)

### 文中所用的强化学习的特点:

* 离线数据还是在线数据
* MDP如何建模: (环境为 用户, agent 是广告推荐系统)
  * 状态 (**这是一个比较强的假设, 用户的状态不止于此**): 
    * 用户在 $t$ 时刻之前的浏览历史
    *  和 用户 $t$ 时刻的请求信息 (上下文信息, 时间, 位置, 推荐列表)
  * 动作
    * 从 `ad candidates` 中选择 一个广告插入到 推荐列表中
  * 奖励
    * 一旦推荐, 可以通过用户的反馈获得一个 immediate reward $r(s_t,a_t)$ , 奖励由两部分构成, 1)广告系统收益, 2)对用户体验的影响
  * 状态转移概率 : (**这是一个比较强的假设, 因为用户的转移概率可能受到很多的场外因素的影响**)
    * $p(s_{t+1}|s_t, a_t)$ 
  * 衰减因子

### 论文中需要关注的问题:

* 解决什么问题
  * 用户体验 与 收益 最优
* 如何解决的
  * 为了解决做了哪些假设? 假设是否合理? 和真实情况是否有出入?
  * 模型
    * 给定一个 recommendation list (内容推荐列表)
    * 模型决定: 1) 是否在里面插入广告, 2) 如果插入,插入的最优位置是哪, 3) 插入的最优广告是啥
    * setting
      * 每个请求会有 $|A|$ 个 ad candidate 整装待发, 推荐内容列表的长度为 $|L|$ 
    * 训练: 纯离线数据训练.
* 为什么能解决
* 使用的一些 trick

### 其它



# SLATEQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets

