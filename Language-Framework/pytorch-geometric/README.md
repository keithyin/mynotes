# 图(数据结构)

**需要关注的点** 图:$<\mathbf V, \mathbf E>$,  $|\mathbf V|$ 表示节点的个数, $|E|$ 表示边的个数

* 图数据的存储: 
  * [矩阵表示](https://blog.csdn.net/Hanging_Gardens/article/details/55670356)
    * 邻接矩阵矩阵表示: $|V|*|V|$ 的矩阵.
    * 关联矩阵表示: $|V|*|E|$ 的矩阵, **每一列**表示一条边, 边的开始值为`-1`, 结束为 `1`.
    * 边表示法: $|E|*2$ 的矩阵, **每一行**表示每条边的开始节点与结束节点.
  * [链表表示](https://www.cnblogs.com/dzkang2011/p/graph_1.html)
    * 邻接表表示法: 一共$|V|$ 个链表, **每个链表**保存着与该节点相邻的节点.
    * [十字链表法](https://blog.csdn.net/dongyanxia1000/article/details/53584496): 邻接表计算出度容易, 但是计算入度困难; 逆邻接表计算入度容易, 但是计算出度困难.



数据结构中的图基本介绍完毕, 下面看 **pytorch-geometric**

# 常见任务

* 图的节点分类
* 小图分类
* 还有其它吗?



# pytorch-geometric 数据表示

* `torch_geometric.data.Data`: 表示**一张图**!!!!!!!!!!

  > 节点的特征是作为 节点的 id 的. 而**不是**通过 `data.x` 中的下标索引值来确定的. 
  >
  > `data.x` 的下标索引值在 `edge_index` 中有作用.
  >
  > `data.x[i]` : 其中 `i` 就是所说的下标索引.

  * `x`: 用来表示节点的特征, 节点特征可以是 `id/one-hot` 来表示自己的 `id`, `[节点个数, 节点特征]` , 节点的特征向量应该存放在 `Embedding` 中.
  * `edge_idx` : 用来表示每条边的 开始节点和结束节点. 
  * `edge_attr` : 用来表示边的属性

```python
class Data(object):
  """
    Args:
        x (Tensor, optional): 节点特征矩阵 :obj:`[num_nodes,num_node_features]`.
        edge_index (LongTensor, optional): 图的边表示法 `[2, num_edges]`.
        edge_attr (Tensor, optional): 边的特征矩阵`[num_edges, num_edge_features]`. 
        y (Tensor, optional): Graph or node targets with arbitrary shape.
            (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        norm (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)

    The data object is not restricted to these attributes and can be extented
    by any other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.norm = norm
        self.face = face
```



### 常用数据集





### 如何mini-batch

* 数据集只有一张大图的情况
* 数据集是多张小图的情况



### 构建数据集

* 提供了两种类型的 `Dataset`: `data.Dataset , data.InMemoryDataset` , 名如其意.
* 



# Message Passing

其实图神经网络计算过程可以作以下总结:

* 通过 **邻接矩阵** 来获取和和 **当前节点** 相邻的节点
* 然后将相邻节点的特征聚集起来, 执行某些计算 (`pooling`, 或者一些其它的操作, 甚至可以 `attention`)
* 计算结果作为**当前节点** 的新向量表示.

数学公式:
$$
\mathbf x_i^{(k)} =\gamma^{(k)}\Bigg(\mathbf x_i^{(k-1)}, ◻_{j\in \mathcal N(i)}\phi^{(k)}\Big(\mathbf x_i^{(k-1)},\mathbf x_j^{(k-1)},\mathbf e_{i,j} \Big)\Bigg)
$$

* $\mathbf x_i^{(k)} \in \mathbb R^n$ : 表示 节点 $i$ 在第 $k$ 层的 `embedding` 表示
* $\mathbf e_{i,j} \in \mathbb R^m$ : 表示节点 $i->j$ 这条边的特征向量
* $\gamma(), 和 \phi()$ : 表示可微的神经网络
* ◻:表示differentiable, permutation invariant function. (函数哦, 不包含参数的); 例如: `sum, max, mean`



### nn.MessagePassing

* 框架干了啥, 我们需要干啥
* 需要继承重写的部分
  * `message` : 表示 $\phi ()$
  * `udpate` : 表示 $\gamma ()$
  * `forward` : 里实现取 `embedding` 的操作是可以的

```python
class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add', flow='source_to_target'):
       pass
      
    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out

```

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
```



## MessagePassing实现细节

* 在 `pytorch-geometric` 里， `edge_index` （节点的连接关系）有两种表示方式， 一个是使用 `Tesnor` 表示，一种是使用 `SparseTensor` 表示。
* 先看 `Tensor` 表示
  * 在使用`Tensor` 表示的时候，`Tensor` 的shape 为 `(2, num_edges)` 即：使用边的连接来表示图，而非邻接矩阵。
  * 在阅读`MessagePassing` 源码的时候，在 `__collect__` 里面
* 我们看gcn论文的时候可以发现， 文章中是使用 邻接矩阵 来表示节点的相互连接关系的。
* 