# 常用算法总结



**Logistic-Regression**

```python
class LogisticRegression(penalty='l2',
                         dual=False,
                         tol=0.0001,
                         C=1.0, # 值越小, 表示约束越强
                         fit_intercept=True,
                         intercept_scaling=1,
                         class_weight=None, # dict or 'balanced'!!
                         random_state=None, 
                         solver='liblinear', 
                         max_iter=100,  # 如果没有收敛, 会有提示的.
                         multi_class='ovr', 
                         verbose=0, 
                         warm_start=False, 
                         n_jobs=1
                        )
"""
classifier = LogisticRegression(solver='sag')

print("fitting ...")

# 无论是 2 分类还是多分类, train_target.shape = [num_samples]
classifier.fit(train_features, train_target)

# [num_samples, num_classes], 类别的排序在 self.classes_ 中.
probas = classifier.predict_proba(test_features)

"""
```



**SVM**

```python
# 分类
class SVC(C=1.0, kernel='rbf', 
          degree=3, 
          gamma='auto', 
          coef0=0.0,
          shrinking=True,
          probability=False,
          tol=0.001,
          cache_size=200,
          class_weight=None, 
          verbose=False,
          max_iter=-1, 
          decision_function_shape='ovr', 
          random_state=None)

"""
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC() # ovr 舒服. predict 的结果就是 [num_samples, num_classes]
clf.fit(X, y) 

print(clf.predict([[-0.8, -1]]))

# 多类分类
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X, Y) 
"""

# 使用 SVM 做回归
class SVR(kernel='rbf', 
          degree=3, 
          gamma='auto', 
          coef0=0.0, 
          tol=0.001,
          C=1.0, 
          epsilon=0.1,
          shrinking=True, 
          cache_size=200,
          verbose=False,
          max_iter=-1)
"""
from sklearn.svm import SVR
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y)
"""
```



**LinearRegression**

```python
class LinearRegression(fit_intercept=True,
                       normalize=False, 
                       copy_X=True, 
                       n_jobs=1)
```

