

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

**增加节点的逻辑**

* 新增节点的 next 包含的是索引类型的最大值，如果索引类型是 i32，那么就是 i32::MAX, 表示 null
* 新增节点放到 Graph.nodes 里

**增加边的逻辑**

* 假设a是边的头节点，b是尾节点
  * `new_edge.next[0] = a.next[0];`
  * `new_edge.next[1] = b.next[1];`
  * `a.next[0] = new_edge;`
  * `b.next[1] = new_edge;`


**节点的所有入边**

假设节点a：

* `e = a.next[0]`, 节点的出边头，
* `e = e.next[0]`. 就能得到所有的出边

**节点的所有出边**

* `e = a.next[1]`，节点入边的头
* `e = e.next[1]`, 遍历得到所有的入边


