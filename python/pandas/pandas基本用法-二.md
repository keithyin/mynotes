# pandas 基本用法二 (DataFrame和Series)



## DataFrame构建方法

**通过构造函数**

```python
class pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
"""
data : numpy ndarray (structured or homogeneous), dict, or DataFrame
Dict can contain Series, arrays, constants, or list-like objects

index : Index or array-like, 表示 row-index
Index to use for resulting frame. Will default to np.arange(n) if no indexing information part of input data and no index provided

columns : Index or array-like, 表示 column-index
Column labels to use for resulting frame. Will default to np.arange(n) if no column labels are provided

dtype : dtype, default None
Data type to force. Only a single dtype is allowed. If None, infer

copy : boolean, default False
Copy data from inputs. Only affects DataFrame / 2d ndarray input
"""
```



**通过工厂函数**

```python
DataFrame.from_records(data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None)
"""
data : ndarray (structured dtype)

index : string, list of fields, array-like, 表示 row-index
Field of array to use as the index, alternately a specific set of input labels to use

exclude : sequence, default None, 要排除哪列
Columns or fields to exclude

columns : sequence, default None, 表示 column-index
Column names to use. If the passed data do not have names associated with them, this argument provides names for the columns. Otherwise this argument indicates the order of the columns in the result (any names not found in the data will become all-NA columns)

coerce_float : boolean, default False
Attempt to convert values of non-string, non-numeric objects (like decimal.Decimal) to floating point, useful for SQL result sets
"""
# 里面的每个元素是一行元素.
data = [["hello", 2, 3], ["world", 2, 1]]

df = pd.DataFrame.from_records(data)
df.to_csv("from_records.csv", index=None)
# 打开文件
"""
0,1,2   //这一行是列索引
hello,2,3
world,2,1
"""
```

```python
DataFrame.from_dict(data, orient='columns', dtype=None)
"""
data : dict
{field1 : array-like, field2: array-like} or {field : dict}

orient : {‘columns’, ‘index’}, default ‘columns’
The “orientation” of the data. If the keys of the passed dict should be the columns of the resulting DataFrame, pass ‘columns’ (default). Otherwise if the keys should be rows, pass ‘index’.

dtype : dtype, default None
Data type to force, otherwise infer
"""

data = {"col1": [1, 2, 3], "col2": [3, 2, 1]}
df = pd.DataFrame.from_dict(data)
df.to_csv("from_dict.csv", index=None)
# 打开文件可以看到
"""
col1,col2
1,3
2,2
3,1
"""
```

```python
DataFrame.from_items(items, columns=None, orient='columns')
"""
这个和 from_dict 差不多, 就是 dict 变成了 [(key, value), ...]
items : sequence of (key, value) pairs
Values should be arrays or Series.

columns : sequence of column labels, optional
Must be passed if orient=’index’.

orient : {‘columns’, ‘index’}, default ‘columns’
The “orientation” of the data. If the keys of the input correspond to column labels, pass ‘columns’ (default). Otherwise if the keys correspond to the index, pass ‘index’.
"""
```



## Series构建方法

当对 `DataFrame` 进行 **一行** 或 **一列** 索引时, 返回的就是 `Series`

```python
type(dataframe["col1"]) --> Series
type(dataframe.loc[0]) --> Series
```



```python
Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)
"""
data : array-like, dict, or scalar value
Contains data stored in Series

index : array-like or Index (1d)
Values must be hashable and have the same length as data. Non-unique index values are allowed. Will default to RangeIndex(len(data)) if not provided. If both a dict and index sequence are used, the index will override the keys found in the dict.

dtype : numpy.dtype or None
If None, dtype will be inferred

copy : boolean, default False
Copy input data
"""
```



