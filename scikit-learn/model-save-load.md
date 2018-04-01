# sk-learn 模型的保存与加载

**使用pickle**

```python
from sklearn import svm
from sklearn import datasets
import pickle

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
s = pickle.dumps(clf)

clf2 = pickle.loads(s)
clf2.predict(X[0:1])
```



**使用joblib**

```python
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl') # 保存到某文件中
```

```python
clf = joblib.load('filename.pkl')  # 加载回来就可以嗨了.
```





**关于 pickle** 

```python
pickle.dump(obj, file, protocol=None) # 将 obj 的二进制表示 写入 文件中
pickle.dumps(obj, protocol=None) # 将obj 的二进制表示 返回
pickle.load(file) # 从打开的文件中加载原始对象
pickle.loads(bytes) #从二进制文中加载对象.
```



```python
import pickle
with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
```



## 参考资料

[https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html)