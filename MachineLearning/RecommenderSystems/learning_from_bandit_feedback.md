bandit feedback常见于推荐系统中（内容推荐/广告推荐）。因为每次eshow，并不能将所有的item展示给用户让其打分，只能将系统选择的item展示给用户，观察用户的反应。这个情况与multi-arm bandit类似，环境只能给出用户选择的arm的reward。

# deep learning with logged bandit feedback

贡献: 给出了 SGD 优化 self-normalized inverse propensity score 目标函数的方法

