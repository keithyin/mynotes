

* Geometric understanding
* Numeric operations






# MIT Linear Algebra



## Ax=0

* 得到 A 的 null space
* 在对 A 进行消元法时，要保证 null space 是不变的（解是不变的）
* 对 A 进行行变换就可以咯

$$
A =
\begin{bmatrix}
1 &2&2&2\\
2&4&6&8\\
3&6&8&10
\end{bmatrix} \\
\rightarrow
\begin{bmatrix}
1 &2&0&-2\\
0&0&1&2\\
0&0&0&0
\end{bmatrix} = U
$$

> 对 A 进行行变换，变换之后可以看出，有两个主元
>
> * 主元的个数 = 矩阵的 rank
>
> 一些概念
>
> * pivot columns： 主元所在的列
> * free columns：除了主元所在的列
>   * 为什么叫 free，因为在求 $UX=0$ 的时候，free columns 的值可以随便选
>   * $x_2, x_4$ 可以随便固定，然后求 $x_1, x_3$ 的值

**算法**

* 行变换 到 阶梯形矩阵
* 确定主元
* 选取 free columns 的值 然后计算 pivot columns 的值
  * 有几个自由变量 就选几次 自由变量的值



**注意：**

* 如果对列进行移动的化，可以发现 pivot columns and rows 是个 单位矩阵

## AX = b

> 可能有解，也可能没有解

* 增广矩阵

$$
A =\begin{bmatrix}1 &2&2&2&b1\\2&4&6&8&b2\\3&6&8&10&b3\end{bmatrix} \\\rightarrow\begin{bmatrix}1 &2&0&-2\\0&0&1&2\\0&0&0&0\end{bmatrix} = U
$$

**如何求特解**

* 将所有的 free variables 设置为 0， 然后求解 $AX=b$ 中 pivot variables 的值，这样就能得到一个特解。



**算法**

* 求一个特解
* 然后求 $AX=0$ 的 `null space`
* 然后加起来就行了



## 线性相关性，基和维数 

* 线性相关与线性无关

> 不会说一个矩阵是 线性相关/线性无关的, 我们会说 一个**向量组**是 线性相关/线性无关 的.
>
> 一个**向量组** 构成一个空间
>
> 一个**向量组** 作为 基



如果 $v_1, v_2, v_n$ 是 $A$ 矩阵的列

* 如果他们是线性无关的, 那么 $A$ 的 Null Space 只有 0 向量.
* 如果他们是线性相关的, 那么 $A$ 的 Null Space 除了 0 向量还有其它向量.



向量组 Span a space (生成一个空间)

* 构成空间包含 向量组的所有线性组合



向量空间的一组基是 一个向量组:

* 向量组独立
* 向量组可以生成 **整个空间**(不是子空间)



rank(A) = 主元的个数 = 列空间的维度



## 矩阵的四个基本子空间

* 列空间, Column Space, (由矩阵的列构成的空间)
  * **列空间的维度** = 列向量组有**几个基** 
* 零空间, Null Space, $Ax=0$, $x$ 的解
* 行空间, Row Space, 矩阵转置的 列空间
* 矩阵转置的 零空间, Left NULL Space



**关系**

* 行空间和列空间 具有相同的维度, **都是 矩阵的 Rank**
* 行变化会影响 列空间, 不会影响行空间



假设:
$$
A=
\begin{bmatrix}
1 &2&3&1\\
1&1&2&1\\
1&2&3&1
\end{bmatrix}
$$
通过行变换可以得到:


$$
\begin{bmatrix}
1&0&1&1\\
0&1&1&0\\
0&0&0&0
\end{bmatrix}
$$

* 前两个向量就 行空间的基(因为行变换不会影响行空间, 只会影响列空间)
* 也可以看出, 有两个 pivot columns. 因为有两个主元(pivot)



**LEFT NULL SPACE**
$$
A^T y = 0 \\
y^TA = 0^T
$$

* 这就是称之为 left null space, 因为可以看作 $y^T$ 在左边



**矩阵也可以是空间, ALL 3*3 matrix 构成空间**

* **子空间是** : 上三角矩阵, 对称矩阵, 对角矩阵




## 矩阵空间，秩1矩阵，和小世界图

* 矩阵空间：由矩阵之间的操作（加减乘除）构成的空间，
  * n维向量可以看作是 n维空间的一个点，所有的n维向量组成了一个空间
  * 矩阵可以看作什么的一个点呢？
* 所有秩为1的矩阵都可以写成，行向量与列向量的乘积



**小世界图**

* 图：{nodes, edges}



## 图和网络

* 图：{nodes, edges}
* incidence matrix: 关联矩阵
  * edges：几个 edges，矩阵就有几行
  * nodes：有几个 nodes，矩阵就有几列
  * 边的起点用 -1 表示，终点用 1 表示
  * 四个 nodes，三条边的 图用矩阵可以这么表示

$$
\begin{bmatrix}
-1&0&1&0\\
0&-1&1&0\\
0&0&1&-1
\end{bmatrix}
$$

* $A^TA$ 结果总是对称矩阵





## 正交向量与子空间

* 向量的正交
* 子空间的正交
* base 的正交

