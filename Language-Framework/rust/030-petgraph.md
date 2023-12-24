

# Graph

```rust

pub struct Graph<N, E, Ty = Directed, Ix = DefaultIx> {
    nodes: Vec<Node<N, Ix>>,
    edges: Vec<Edge<E, Ix>>,
    ty: PhantomData<Ty>,
}

pub struct Node<N, Ix = DefaultIx> {
    /// Associated node data.
    pub weight: N,
    /// Next edge in outgoing and incoming edge lists.
    next: [EdgeIndex<Ix>; 2],
}


pub struct Edge<E, Ix = DefaultIx> {
    /// Associated edge data.
    pub weight: E,
    /// Next edge in outgoing and incoming edge lists.
    next: [EdgeIndex<Ix>; 2],
    /// Start and End node index
    node: [NodeIndex<Ix>; 2],
}

// 用来表示 node 或者 edge 在 nodes、edges 中索引的变量。
pub struct NodeIndex<Ix = DefaultIx>(Ix);
pub struct EdgeIndex<Ix = DefaultIx>(Ix);

pub type DefaultIx = u32;
```

## 有向图

```rust
pub struct Node<N, Ix = DefaultIx> {
    /// Associated node data.  就是Node的属性信息
    pub weight: N,
    /// next[0] 出边的链表头，next[1] 入边的链表头
    next: [EdgeIndex<Ix>; 2],
}

pub struct Edge<E, Ix = DefaultIx> {
    /// Associated edge data.
    pub weight: E,
    /// Next edge in outgoing and incoming edge lists.
    /// next[0]: 该边的头结点的 下一个 出边
    /// next[1]: 该边的尾节点的 下一个 入边
    next: [EdgeIndex<Ix>; 2],
    /// Start and End node index
    node: [NodeIndex<Ix>; 2],
}

```

