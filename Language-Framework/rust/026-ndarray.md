
# ArrayBase

> The array is a general container of elements

```rust
pub struct ArrayBase<S, D> 
where
    S: RawData, 
{ /* private fields */ }
```

* S: 数据容器, DataContainer
* D: 维度

类型别名：`Array`, `ArcArray`, `CowArray`, `ArrayView`, and `ArrayViewMut`, 都是ArrayBase, 只是data container不一样。

> Type aliases `Array`, `ArcArray`, `CowArray`, `ArrayView`, and `ArrayViewMut` refer to `ArrayBase` with different types for the data
> container: arrays with different kinds of ownership or different kinds of array views.

# Array
> Owned array

```rust
// A: element type
// D: 数据维度
pub struct Array<A, D>


use ndarray::Array3; // Array<A, 3>.
let mut temperature = Array3::<f64>::zeros((3, 4, 5));
// Increase the temperature in this location
temperature[[2, 2, 2]] += 0.5;
```


# ArcArray
> Owned array with reference counted data (shared ownership)

Sharing requires that it uses copy-on-write for mutable operations. Calling a method for mutating elements on ArcArray, for example view_mut() or get_mut(), will break sharing and require a clone of the data (if it is not uniquely held).

# CowArray
> It can represent either an immutable view or a uniquely owned array. If a CowArray instance is the immutable view variant, then
> calling a method for mutating elements in the array will cause it to be converted into the owned variant (by cloning all the
> elements) before the modification is performed.


# ArrayViews

* ArrayView: read-only view
* ArrayViewMut: read-write view

```rust
// 如何构建一个view

.view()
.view_mut()

.slice()
.slice_mut()

// 或者通过一些iter来构建view
```

# 索引与维度

```rust
use ndarray::Array2;
let mut array = Array2::zeros((4, 3));
array[[1, 1]] = 7;

.iter()
.iter_mut() // 都是从right-most维度进行遍历，因为那个维度是内存连续的，遍历比较快
```

# 遍历

```rust
.iter()
.iter_mut() // 都是从right-most维度进行遍历，因为那个维度是内存连续的，遍历比较快

.outer_iter() // 第一位迭代
.axis_iter() // 指定axis进行迭代

.rows()
.columns()
.lanes()

```

# slicing

```rust
.slice()
.slice_mut()
.slice_move()
.slice_collapse()


use ndarray::{arr2, arr3, s, ArrayBase, DataMut, Dimension, NewAxis, Slice};

// 2 submatrices of 2 rows with 3 elements per row, means a shape of `[2, 2, 3]`.

let a = arr3(&[[[ 1,  2,  3],     // -- 2 rows  \_
                [ 4,  5,  6]],    // --         /
               [[ 7,  8,  9],     //            \_ 2 submatrices
                [10, 11, 12]]]);  //            /
//  3 columns ..../.../.../

assert_eq!(a.shape(), &[2, 2, 3]);

// Let’s create a slice with
//
// - Both of the submatrices of the greatest dimension: `..`
// - Only the first row in each submatrix: `0..1`
// - Every element in each row: `..`

let b = a.slice(s![.., 0..1, ..]);
let c = arr3(&[[[ 1,  2,  3]],
               [[ 7,  8,  9]]]);
assert_eq!(b, c);
assert_eq!(b.shape(), &[2, 1, 3]);

// Let’s create a slice with
//
// - Both submatrices of the greatest dimension: `..`
// - The last row in each submatrix: `-1..`
// - Row elements in reverse order: `..;-1`
let d = a.slice(s![.., -1.., ..;-1]);
let e = arr3(&[[[ 6,  5,  4]],
               [[12, 11, 10]]]);
assert_eq!(d, e);
assert_eq!(d.shape(), &[2, 1, 3]);

// Let’s create a slice while selecting a subview and inserting a new axis with
//
// - Both submatrices of the greatest dimension: `..`
// - The last row in each submatrix, removing that axis: `-1`
// - Row elements in reverse order: `..;-1`
// - A new axis at the end.
let f = a.slice(s![.., -1, ..;-1, NewAxis]);
let g = arr3(&[[ [6],  [5],  [4]],
               [[12], [11], [10]]]);
assert_eq!(f, g);
assert_eq!(f.shape(), &[2, 3, 1]);

// Let's take two disjoint, mutable slices of a matrix with
//
// - One containing all the even-index columns in the matrix
// - One containing all the odd-index columns in the matrix
let mut h = arr2(&[[0, 1, 2, 3],
                   [4, 5, 6, 7]]);
let (s0, s1) = h.multi_slice_mut((s![.., ..;2], s![.., 1..;2]));
let i = arr2(&[[0, 2],
               [4, 6]]);
let j = arr2(&[[1, 3],
               [5, 7]]);
assert_eq!(s0, i);
assert_eq!(s1, j);

// Generic function which assigns the specified value to the elements which
// have indices in the lower half along all axes.
fn fill_lower<S, D>(arr: &mut ArrayBase<S, D>, x: S::Elem)
where
    S: DataMut,
    S::Elem: Clone,
    D: Dimension,
{
    arr.slice_each_axis_mut(|ax| Slice::from(0..ax.len / 2)).fill(x);
}
fill_lower(&mut h, 9);
let k = arr2(&[[9, 9, 2, 3],
               [4, 5, 6, 7]]);
assert_eq!(h, k);

```


# 数学计算

符号说明：

* A：array, array_view都可以
* B: array with owned storage (Array 或 ArcArray)
* C: array with mutable data (Array, ArcArray 或 ArrayViewMut)
* @: 表示任意 二元操作符 (+, -, *, / 等等)


* &A @ &A which produces a new Array
* B @ A which consumes B, updates it with the result, and returns it
* B @ &A which consumes B, updates it with the result, and returns it
* C @= &A which performs an arithmetic operation in place

```rust
use ndarray::{array, ArrayView1};

let owned1 = array![1, 2];
let owned2 = array![3, 4];
let view1 = ArrayView1::from(&[5, 6]);
let view2 = ArrayView1::from(&[7, 8]);
let mut mutable = array![9, 10];

let sum1 = &view1 + &view2;   // Allocates a new array. Note the explicit `&`.
// let sum2 = view1 + &view2; // This doesn't work because `view1` is not an owned array.
let sum3 = owned1 + view1;    // Consumes `owned1`, updates it, and returns it.
let sum4 = owned2 + &view2;   // Consumes `owned2`, updates it, and returns it.
mutable += &view2;            // Updates `mutable` in-place.
```

**Array与Scalar操作**

* &A @ K or K @ &A which produces a new Array
* B @ K or K @ B which consumes B, updates it with the result and returns it
* C @= K which performs an arithmetic operation in place


**一元操作符**

* A：array 和 arrayview 都可以
* B：an array with owned storage (either Array or ArcArray)
* @: !, -

* @&A which produces a new Array
* @B which consumes B, updates it with the result, and returns it



# 参考资料

* https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#array
* https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html
