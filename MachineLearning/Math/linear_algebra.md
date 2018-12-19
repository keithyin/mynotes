

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

* 从空间的角度理解：当 $b$ 处于由 $A$ 的列空间中的时候，会有解
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





## 14：正交向量与子空间

* 正交，垂直，90



* 向量的正交
  * dot product 结果为 0， $x^Ty=0$
  * 零向量和所有向量正交
  * 当垂直的时候，下面的等式成立，可以得到 $x^Ty=0$
  * $||x||^2 = x^Tx$

$$
||x||^2+||y||^2 = ||x+y||^2
$$



* 子空间的正交， S 与 T 正交
  * S 中的 所有向量 与 T 中的所有向量正交
  * 如果两个空间相交，那么他们一定不正交，除非相交与 0 点
  * 由于 $Ax=0$ ，所以
    * row space 和 null space 正交
      * null space 包含 **所有** 与 row space 正交的向量
    * column space 和 $A^T$ 的 null space 正交
* base 的正交
  * ​



**当 $Ax=b$ 没有解的时候，如何求解 $Ax=b$，在应用中非常常见， 因为会有脏数据出现**

* 重点是掌握 $A^TA$， 具有以下属性
  * 方阵
  * 对称

$$
Ax=b \rightarrow A^TA\hat x = A^Tb
$$

* 希望 $\hat x$ 的值就是 $Ax=b$ 的最优解

$$
nullspace(A^TA) = nullspace(A) \\
rank(A^TA) = rank(A)
$$



## 15：子空间投影

* 投影，将向量投影到另外 一个 子空间中。
* 投影矩阵
  * 投影到一维子空间的话： $P = aa^T / (a^Ta)$ 为投影矩阵，$Pb$  就是投影的结果
  * 投影矩阵是 对称的， $P^T=P$
  * 对于投影矩阵：$P^2 = P$ ，因为投影两次依旧会落到同一个点



**为什么要 projection**

* 因为 $Ax=b$ 可能没有解？所以只能求解 **最接近的问题** 的解。
* 所以我们可以微调 $b$ ,将 $b$ 微调到 $A$ 的列空间中。
* 可以通过投影的方式将 $b$ 映射到 $A$ 的列空间中。
* 所以 $Ax=b$ 问题变成，$A\hat x = p$,    $p$ 为 $b$ 在 A 的列空间上的投影
* 这时，$b-A\hat x$ 就与 A 的 **列空间垂直**，所以可以得到 $A^T(b-A\hat x)=0$
* 再化简得到 $A^TA\hat x=A^Tb$ ，得到等式



**分析 $A^TA\hat x=A^Tb$**

* $\hat x = (A^TA)^{-1}A^Tb$
* $p=A\hat x = A(A^TA)^{-1}A^Tb$ ，所以将 $b$ 投影到 $A$ 的列空间的 投影矩阵是 $A(A^TA)^{-1}A^T$



**最小二乘**

* 方程组个数远多余未知数，可以保证 $A^TA$ 可逆
* 方程式无解，但是可以求出最优解。使用 $A^TA\hat x=A^Tb$ ，求出来的 $\hat x$ 就是最优解





##  16：投影矩阵和最小二乘

**投影矩阵**
$$
P=A(A^TA)^{-1}A^T
$$

* $I-P$ 也是一个投影矩阵



* 如果 A has independent  columns, 那么 $A^TA$ 就是可逆的
* 当矩阵可逆的话，那么 $Ax=0$ 只有 0解。
* ​





## 17：正交矩阵和Gram-Schmidt正交化

* 标准正交：垂直 且 单位向量
* 正交基  
* 正交矩阵
  * $Q^TQ=I$

$$
q_i^Tq_j = 
\begin{cases}
0, &    i \neq j\\
1, &  i = j
\end{cases}
$$



$Q= [q_1, q_2, q_3, ...]$

* 为什么需要 $Q$ 呢？ 有了 $Q$ 之后，很多操作变的简单了呢？
  * 如果 $Q$ 的拥有 orthonormal columns 
  * 那么 投影矩阵就会变成：$P=Q(Q^TQ)^{-1}Q^T = QQ^T$ ($Q$ 不需要是方阵)



**Gram-Schmidt**

* 使列向量正交，并且 normal
* 算法
  * 固定一个向量
  * 第二个向量使用 投影 的垂直边 决定第二个向量的方向（减去第一个向量方向上的分量）
  * 第三个向量垂直与前两个向量构成的空间，也是使用投影求解出来（减去第一个向量和第二个向量上的分量。）
* 正交化之后，还是代表同一个空间



**$A=QR$, 矩阵 A 和 正交化矩阵的 Q 的关系**

* $R$ 是上三角矩阵，因为构造出来的向量，是垂直于前面的向量的。



##  18：行列式及其性质

**方阵的两个重点**

* 行列式
* 特征值与特征向量



**行列式的几个性质, 方阵 A**

* 单位阵的 行列式 为 1
* 交换矩阵的 行，reverse sign of the det
* 第三个性质，下面两个公式

$$
det \begin{bmatrix}
ta&tb\\
c&d\\
\end{bmatrix} = t *det\begin{bmatrix}
a&b\\
c&d\\
\end{bmatrix}
$$

$$
det \begin{bmatrix}
a+a'&b+b'\\
c&d\\
\end{bmatrix} = det\begin{bmatrix}
a&b\\
c&d\\
\end{bmatrix}+
det\begin{bmatrix}
a'&b'\\
c&d\\
\end{bmatrix}
$$

* 如果有两行是线性相关的的，则行列式为0
* 从 $row_k$ 减去 $l * row_i$ ，行列式不变，从第三个性质和第四个性质可以得到
  * 即：初等行变换不会更改 行列式的值
* 三角矩阵的 det 是对角线上的值的乘积
* $det A=0$ 说明， $A$ 是奇异矩阵（singular matrix）
* $det AB = det A * det B$
* $det A^T = det A$ , 这个性质可以说明，对列进行操作 享受 对行操作 一样的性质



##  20：克拉默法则、逆矩阵、体积

**克拉默法则**

* ​



## 21：特征值和特征向量

> 方阵 ， 不需要考虑可逆



**特征向量**

* $Ax = \lambda x$ , $Ax$ 与 $x$ 平行，这时 $x$ 就是特征向量， $\lambda$ 就是特征值



**特征值的性质**

* n*n 的矩阵有 n 个特征值
* 特征值的和 = $trace(A)$
* 特征值的积是 $det (A)$
* $A+nI$：特征值 + n，特征向量不变
* ​



**算法，求解 特征值和特征向量**

* $(A-\lambda I)x = 0$
* 另 $det (A-\lambda I)=0$ 求 $\lambda$ 的值
  * 因为只有 行列式为 0， 上式对于 $x$ 才有非零解。
* 再将 $\lambda$ 带入 求对应的 $x$ 的值。（使用消元法，求 null space。）



## 22：对角化和A的幂

**对角化一个矩阵**

* $S^{-1}AS=\Lambda$



**对角化**

* 当 A 有 n 个线性无关的特征向量
* 将他们按列放组成 S

$$
AS = A [x_1, x_2, ..., x_n] = [\lambda_1x_1, \lambda_2x_2,..., \lambda_nx_n]=[x_1, x_2, ..., x_n] diag(\lambda_1, ..., \lambda_n) = S\Lambda
$$



**特征值的作用**

* 可以用来得到 矩阵幂 的信息。



## 26： 对称矩阵及正定性

> 对称矩阵，非常重要的一种矩阵

* 特征值：特征值是 real
* 特征向量：互相垂直的（如果碰到有相等的特征值，那就取垂直的正交向量）
* 对于对称矩阵来说，正主元的个数=正特征值的个数，负主元的个数=负特征值的个数
* 对于对称矩阵来说，主元的乘积=特征值的乘积=$det(A)$



**通常情况下：**
$$
A=S\Lambda S^{-1}
$$
**如果A是对称矩阵**
$$
A=S\Lambda S^{-1} = Q\Lambda Q^{-1} = Q\Lambda Q^T
$$
**对于标准正交矩阵来说**
$$
Q^{-1} = Q^T
$$




**对称矩阵**

* positive definite matrix:
  * 所有的 **特征值** 都是正的
  * 所有的 **主元** 都是正的
  * 所有的 **子矩阵的行列式** 都是正的
* 微分方程：知道特征值的正负性，就知道方程的稳定性



## 28： 正定矩阵和最小值

> 在讨论正定性的时候：矩阵首先是 对称的

* 如何判断一个矩阵是不是 positive definite matrix
  * 所有的 **特征值** 都是正的
  * 所有的 **主元** 都是正的
  * 所有的 **子矩阵的行列式** 都是正的
  * $x^TAx>0$ , 对于任何 $x \neq 0$都成立
* positive semi-definite



**消元法的本质就是配方法，可以找个 A 和 $x^TAx$ 试一下**



**$A^TA$  是正定矩阵**



在微积分中，判断一个函数的最小值需要两个条件

* 某个点的一阶导为 0
* 二阶导是正的

如果是多元的话，就上升到矩阵了

* 一阶导为 0
* 二阶导矩阵是 正定的。



## 29： 相似矩阵和若尔当形

> 方阵

* 相似矩阵， A 和 B 相似, 则存在一个 矩阵 $M$
  * $B = M^{-1}AM$
* 可以发现 $S^{-1}AS=\Lambda$
* 相似矩阵具有相同的 特征值



## 30：  奇异值分解

$$
A = U\Sigma V^T
$$

* 假定  $m\ge n$
* A 可以是任何矩阵, $m*n$
* $\Sigma$ 是个对角阵 , ($n*n$)
* $U, V$ 都是单位正交矩阵， （$U (m*n)$ 列空间的标准正交基，$V (n*n)$ 行空间的标准正交基）
* 当 A 是正定的时候 $A=Q\Lambda Q^T$ , 一个正交矩阵就能解决问题
* $Av_i = \sigma_i u_i$




**奇异值分解**

* 行空间的一组正交基，经过变换，得到列空间的一组正交基。要得到这个目的，公式可以写成以下形式。 $A[v1, v2, ..., vr] = [u1, u2, ..., ur]diag(\sigma1, \sigma2, ..., \sigma r)$

* 如何计算？ 只要有长方形矩阵存在 $A^TA$ 就是个好东西

$$
\begin{aligned}
A^TA&=(U\Sigma V^T)^T (U\Sigma V^T)\\
&=V\Sigma^TU^TU\Sigma V^T \\
& = V \Sigma^2V^T
\end{aligned}
$$

* $V$ 是 $A^TA$ 的特征向量
* $U$ 是 $AA^T$ 的特征向量



**$Ax=b$ 的几何解释**

* 可以这么看： $A$ 为坐标系的点 $x$ 在标准坐标系上的坐标为 $b$
* 可以这么看： $x$ 对 $A$ 进行列变换，得到的依旧是 $A$ 列空间中的点
* ​



**参考资料**

* [https://medium.com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-svd-162ac796c27d](https://medium.com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-svd-162ac796c27d)
* [https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254](https://blog.statsbot.co/singular-value-decomposition-tutorial-52c695315254)
* ​

## 31： 线性变换及对应矩阵



## 32：  基变换和图像压缩



## 34：  左右逆和伪逆





# 消元法

* 求方程组的值
* 求行列式



# 矩阵分解

* $A=LU$ , 消元法
* $A=QR$ ，正交化
* $A=S\Lambda S^{-1}$  ，对角化



# 矩阵的一些值

* 主元值： 消元法
* 特征值： 求特征值和特征向量