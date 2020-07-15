# 图(数据结构)

**需要关注的点** 图:$<\mathbf V, \mathbf E>$,  $|\mathbf V|$ 表示节点的个数, $|E|$ 表示边的个数

* 图数据的存储: 
  * [矩阵表示](https://blog.csdn.net/Hanging_Gardens/article/details/55670356)
    * 邻接矩阵矩阵表示: $|V|*|V|$ 的矩阵.
    * 关联矩阵表示: $|V|*|E|$ 的矩阵, **每一列**表示一条边, 边的开始值为`-1`, 结束为 `1`.
    * 边表示法: $|E|*2$ 的矩阵, **每一行**表示每条边的开始节点与结束节点.
  * [邻接表](https://www.cnblogs.com/dzkang2011/p/graph_1.html)
    * 邻接表表示法: 一共$|V|$ 个链表, **每个链表**保存着与该节点相邻的节点.
    * [十字链表法](https://blog.csdn.net/dongyanxia1000/article/details/53584496): 邻接表计算出度容易, 但是计算入度困难; 逆邻接表计算入度容易, 但是计算出度困难.
  * 在实际应用中，因为图比较稀疏，所有常用邻接表表示，但是邻接表的长度不一，很难构成一个`tensor`，但是目前的深度学习又是 `tensor` 计算的天下，所以在深度学习中比较好的表示方法就是**使用边** 来表示了。这样很容易可以构建一个 `[2, num_edges]` 的 `tensor`



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
    * 这个在`pytorch-geo` 中可以使用 形状为 `[2, num_edges] ` 的  `tensor` 表示，
    * 也可以使用`SparseTensor` 表示。但是仔细考虑一下，这两种方式殊途同归。使用`SparseTensor` 来表示邻接矩阵的话，那还是 需要两个参数表示，一个是 指定位置，一个用来指定位置所对应的值是多少，指定位置的 `tensor` 就是边。
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

* 现在阅读 `MessagePassing` 源码

* `MessagePassing` 主要的是 `propagate` 代码，这里面走了 `MessagePassing` 的整个流程
  * 输入数据处理
  *  `message`：对消息进行处理（这里指的消息是，source  to target 的消息(每个边上的embedding)）
  * `aggregate`: 如何对消息进行聚合，求均值，最小值，最大值？ 还是 attention？如果是attention的话，是要自己手动撸代码咯。
  * `update`: 消息聚合完之后还需要啥其他的操作？可以在这个方法中实现。
  * 以上的三个方法是我们可以自定义的。
  * ps：`message`  和  `aggregate` 可以整到一起 ，那就是需要实现方法  `message_and_aggregate`  了。
  * ps2：因为代码中大量使用 `kwargs` 所以看起来有点迷，这里总结一下。因为`MessagePassing` 的`propagate` 其实是一个模板模式，里面已经规定好了调用 `message, aggregate, update` 。 如果 `message, aggregate, update`  需要传额外的参数，直接通过 `propagete` 传进去就好了。如何传呢？名字对应，`kwargs` 会搞定一切。。。`message` 的参数需要额外注意一下！

```python
class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers of the form
    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"` or :obj:`None`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`-2`)
    """

    special_args: Set[str] = set([
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size_i',
        'size_j', 'ptr', 'index', 'dim_size'
    ])

    def __init__(self, aggr: Optional[str] = "add",
                 flow: str = "source_to_target", node_dim: int = -2):

        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max', None]

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.inspector.inspect(self.aggregate, pop_first=True)
        self.inspector.inspect(self.message_and_aggregate, pop_first=True)
        self.inspector.inspect(self.update, pop_first=True)
		
        # message， aggregate， update 这些函数形参与 special_args 的diff 才是用户传进来的参数！
        self.__user_args__ = self.inspector.keys(
            ['message', 'aggregate', 'update']).difference(self.special_args)
        self.__fused_user_args__ = self.inspector.keys(
            ['message_and_aggregate', 'update']).difference(self.special_args)

        # Support for "fused" message passing. 
        # 通过判断子类是否实现了 message_and_aggregate 方法来判断
        self.fuse = self.inspector.implements('message_and_aggregate')

        # Support for GNNExplainer.
        self.__explain__ = False
        self.__edge_mask__ = None

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            assert edge_index.dtype == torch.long
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                (f'Encountered tensor with size {src.size(self.node_dim)} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        """
        args： 这里传入的是 self.__user_args__ 或者 self.__fused_user_args__
        
        edge_index 是一个 shape 为 `[2, num_edges]` 的 tensor
        	pytorch-geo 认为 edge_index[0, num_edges]  是 source
        	edge_index[1, num_edges] 为 taget
        	代码实现上，pytorch-geo 认为 i 是 target，通过将 0或者1 赋值给 i 来决定哪个是 target
        """
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                """
                这部分代码看的贼难受， 为什么会用 形参 key 的后缀来决定 dim ？？？
                这部分迷的话 可以看 gcn 代码的 propagate 和 message 的形参
                """
                dim = 0 if arg[-2:] == '_j' else 1
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    """
                    lift干了啥：
                    	我们传进来的 data 是个 shape为[num_nodes, feat_size] 的一个 tensor
                    	这里根据 edge_index 将其搞成 [num_edges, feat_size] 的一个 tensor
                    	即：将 node 的 embedding 搞到 message passing 的边上去
                    	lift 之后的 data 每行表示的就是 edge_index 中的 的每条边的 embedding
                    为啥要这样操作：
                    	便于后续的 aggregate 计算。
                    	只用 out['index'] 执行 scatter 操作，边上的 embedding 就乖乖的跑到 对应的target node上去了。                  
                    """
                    data = self.__lift__(data, edge_index,
                                         j if arg[-2:] == '_j' else i)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None
        elif isinstance(edge_index, SparseTensor):
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_weight'] = edge_index.storage.value()
            out['edge_attr'] = edge_index.storage.value()
            out['edge_type'] = edge_index.storage.value()

        out['index'] = out['edge_index_i']
        out['size_i'] = size[1] or size[0]
        out['size_j'] = size[0] or size[1]
        out['dim_size'] = out['size_i']

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            # 注意这里对于 self.message 的调用，传给 propagete的 message的参数会传给 message。
            # 这里的 参数名字 是有一一对应的关系的。唯一一点不同的就是 message x_j/x_i 对应的其实是 propagete 中的 x
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(0) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(0) == edge_mask.size(0)
                out = out * edge_mask.view(-1, 1)

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)
     
    def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs

    @torch.jit.unused
    def jittable(self, typing: Optional[str] = None):
        r"""Analyzes the :class:`MessagePassing` instance and produces a new
        jittable module.

        Args:
            typing (string, optional): If given, will generate a concrete
                instance with :meth:`forward` types based on :obj:`typing`,
                *e.g.*: :obj:`"(Tensor, Optional[Tensor]) -> Tensor"`.
        """
        # Find and parse `propagate()` types to format `{arg1: type1, ...}`.
        if hasattr(self, 'propagate_type'):
            prop_types = {
                k: sanitize(str(v))
                for k, v in self.propagate_type.items()
            }
        else:
            source = inspect.getsource(self.__class__)
            match = re.search(r'#\s*propagate_type:\s*\((.*)\)', source)
            if match is None:
                raise TypeError(
                    'TorchScript support requires the definition of the types '
                    'passed to `propagate()`. Please specificy them via\n\n'
                    'propagate_type = {"arg1": type1, "arg2": type2, ... }\n\n'
                    'or via\n\n'
                    '# propagate_type: (arg1: type1, arg2: type2, ...)\n\n'
                    'inside the `MessagePassing` module.')
            prop_types = split_types_repr(match.group(1))
            prop_types = dict([re.split(r'\s*:\s*', t) for t in prop_types])

        # Parse `__collect__()` types to format `{arg:1, type1, ...}`.
        collect_types = self.inspector.types(
            ['message', 'aggregate', 'update'])

        # Collect `forward()` header, body and @overload types.
        forward_types = parse_types(self.forward)
        forward_types = [resolve_types(*types) for types in forward_types]
        forward_types = list(chain.from_iterable(forward_types))

        keep_annotation = len(forward_types) < 2
        forward_header = func_header_repr(self.forward, keep_annotation)
        forward_body = func_body_repr(self.forward, keep_annotation)

        if keep_annotation:
            forward_types = []
        elif typing is not None:
            forward_types = []
            forward_body = 8 * ' ' + f'# type: {typing}\n{forward_body}'

        root = os.path.dirname(osp.realpath(__file__))
        with open(osp.join(root, 'message_passing.jinja'), 'r') as f:
            template = Template(f.read())

        uid = uuid1().hex[:6]
        cls_name = f'{self.__class__.__name__}Jittable_{uid}'
        jit_module_repr = template.render(
            uid=uid,
            module=str(self.__class__.__module__),
            cls_name=cls_name,
            parent_cls_name=self.__class__.__name__,
            prop_types=prop_types,
            collect_types=collect_types,
            user_args=self.__user_args__,
            forward_header=forward_header,
            forward_types=forward_types,
            forward_body=forward_body,
            msg_args=self.inspector.keys(['message']),
            aggr_args=self.inspector.keys(['aggregate']),
            msg_and_aggr_args=self.inspector.keys(['message_and_aggregate']),
            update_args=self.inspector.keys(['update']),
            check_input=inspect.getsource(self.__check_input__)[:-1],
            lift=inspect.getsource(self.__lift__)[:-1],
        )

        # Instantiate a class from the rendered JIT module representation.
        cls = class_from_module_repr(cls_name, jit_module_repr)
        module = cls.__new__(cls)
        module.__dict__ = self.__dict__.copy()
        module.jittable = None

        return module

```



