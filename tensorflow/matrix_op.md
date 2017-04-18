矩阵操作
```python
#对于2-D
#所有的reduce_...，如果不加axis的话，都是对整个矩阵进行运算
tf.reduce_sum(a, 1） #对axis1
tf.reduce_mean(a,0) #每列均值
#第二个参数是axis，如果为0的话，对其进行求和[i,:]， 如果是1的话，对[:,i]进行求和
#NOTE:返回的都是行向量

#关于concat，可以用来进行降维 3D->2D , 2D->1D
tf.concat(concat_dim, data)
#如果concat为0,就是将第axis=0 的数据连接起来[i,:,:]
#如果为1, 就把axis=1的数据连起来[:,i,:]
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

#squeeze 降维 维度为1的降掉
tf.squeeze(arr, [])
降维， 将维度为1 的降掉
arr = tf.Variable(tf.truncated_normal([3,4,1,6,1], stddev=0.1))
arr2 = tf.squeeze(arr, [2,4])
arr3 = tf.squeeze(arr) #降掉所以是1的维

#split
tf.split(split_dim, num_split, value, name='split')
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(1, 3, value)
tf.shape(split0) ==> [5, 10]

#embedding
mat = np.array([1,2,3,4,5,6,7,8,9]).reshape((3,-1))
ids = [[1,2], [0,1]]
res = tf.nn.embedding_lookup(mat, ids)
res.eval()
array([[[4, 5, 6],
        [7, 8, 9]],

       [[1, 2, 3],
        [4, 5, 6]]])

#扩展维度，如果想用广播特性的话，经常会用到这个函数
# 't' is a tensor of shape [2]
#一次扩展一维
shape(tf.expand_dims(t, 0)) ==> [1, 2]
shape(tf.expand_dims(t, 1)) ==> [2, 1]
shape(tf.expand_dims(t, -1)) ==> [2, 1]
# 't2' is a tensor of shape [2, 3, 5]
shape(tf.expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(tf.expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(tf.expand_dims(t2, 3)) ==> [2, 3, 5, 1]

```