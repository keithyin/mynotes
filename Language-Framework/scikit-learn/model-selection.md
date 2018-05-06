# 如何使用 scikit-learn 做模型选择



**交叉验证**

```python
from sklearn.model_selection import cross_val_score


# 返回交叉验证的 loss
cv_losses = cross_val_score(classifier, train_features, train_target, cv=5, n_jobs=-1)

```



```python
from sklearn import preprocessing

# 划分一下数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=0)
```





## 参考资料

[http://scikit-learn.org/stable/modules/cross_validation.html](http://scikit-learn.org/stable/modules/cross_validation.html)

