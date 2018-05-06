# tensorflow collection
`tensorflow`的`collection`提供一个全局的存储机制，不会受到`变量名`生存空间的影响。一处保存，到处可取。

## 接口介绍
```python
#向collection中存数据
tf.Graph.add_to_collection(name, value)

#Stores value in the collection with the given name.
#Note that collections are not sets, so it is possible to add a value to a collection
#several times.
# 注意，一个‘name’下，可以存很多值

tf.add_to_collection(name, value)
#这个和上面函数功能上没有区别，区别是，这个函数是给默认图使用的
```

```python
#从collection中获取数据
tf.Graph.get_collection(name, scope=None)

Returns a list of values in the collection with the given name.

This is different from get_collection_ref() which always returns the actual
collection list if it exists in that it returns a new list each time it is called.

Args:

name: The key for the collection. For example, the GraphKeys class contains many
standard names for collections.
scope: (Optional.) If supplied, the resulting list is filtered to include only
items whose name attribute matches using re.match. Items without a name attribute
are never returned if a scope is supplied and the choice or re.match means that
a scope without special tokens filters by prefix.
#返回re.match(r"scope", item.name)匹配成功的item, re.match（从字符串的开始匹配一个模式）
Returns:

The list of values in the collection with the given name, or an empty list if no
value has been added to that collection. The list contains the values in the
order under which they were collected.
```
## 思考
`tf`自己也维护一些`collection`，就像我们定义的所有`summary op`都会保存在`name=tf.GraphKeys.SUMMARIES`。这样，`tf.get_collection(tf.GraphKeys.SUMMARIES)`就会返回所有定义的`summary op`





**参考资料**
[https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph.add_to_collection](https://www.tensorflow.org/api_docs/python/framework/core_graph_data_structures#Graph.add_to_collection)
