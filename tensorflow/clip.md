# tensorflow Clip
`tensorflow` 提供了多个关于`clip` 的 `op`, 下面来看一下它们各自都有些什么样的行为.

## tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
将传入的`Tensor t`中的每个元素`clip` 到 `[clip_value_min, clip_value_max]`范围内.

## tf.clip_by_norm(t, clip_norm, axes=None, name=None)

* t : `Tensor`
* clip_norm: 最大 `L2-norm`值
* axes: 指定对哪些轴进行`L2-norm`计算,默认是所有的轴
> l2-norm : sqrt(sum(x**2))


> Specifically, in the default case where all dimensions are used for calculation, if the L2-norm of t is already less than or equal to clip_norm, then t is not modified. If the L2-norm is greater than clip_norm, then this operation returns a tensor of the same type and shape as t with its values set to:
`t * clip_norm / l2norm(t)`
In this case, the L2-norm of the output tensor is clip_norm.
As another example, if t is a matrix and axes == [1], then each row of the output will have L2-norm equal to clip_norm. If axes == [0] instead, each column of the output will be clipped.

## tf.clip_by_average_norm(t, clip_norm, name=None)
和上边干的活应该差不多,不过这个`average`让人很费解.

## clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)

> t_list[i] * clip_norm / max(global_norm, clip_norm)
global_norm = sqrt(sum([l2norm(t)** 2 for t in t_list]))

* t_list: A tuple or list of mixed Tensors, IndexedSlices, or None.

* clip_norm: A 0-D (scalar) Tensor > 0. The clipping ratio.

* use_norm: 如果提供了这个值,就用这个值当作`global norm`,如果不提供的话,就用`global_norm()`来计算

* name: A name for the operation (optional).

返回:
* list_clipped: A list of Tensors of the same type as list_t.
* global_norm: A 0-D (scalar) Tensor representing the global norm.
