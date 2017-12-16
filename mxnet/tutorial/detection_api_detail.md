# mxnet 目标检测 API 介绍

mxnet 的 contrib 中有一些目标检测可用的 API, 用这些 API 可以大大减小我们编写目标检测代码的难度，他们分别是：

* [MultiBoxPrior ](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html?highlight=multiboxprior#mxnet.ndarray.contrib.MultiBoxPrior)
* [MultiBoxTarget](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html?highlight=multiboxprior#mxnet.ndarray.contrib.MultiBoxTarget)
* [MultiBoxDetection](https://mxnet.incubator.apache.org/api/python/ndarray/contrib.html?highlight=multiboxprior#mxnet.ndarray.contrib.MultiBoxDetection)
* ......



## MultiBoxPrior

**用来生成 anchor, 返回的 anchor 是 `[1, height*width*num_anchors_per_cell, 4]`, 4 代表的是 `(xmin, ymin, xmax, ymax)` ** ， 坐标是被归一化了的。



`mxnet.ndarray.contrib.MultiBoxPrior(data=None, sizes=_Null, ratios=_Null, clip=_Null, steps=_Null, offsets=_Null, out=None, name=None, **kwargs)`



* `data`: 要做 anchor 的 feature map（四维）。这个输入进去其实就只是用了它的 高宽两个属性。！
* `sizes` :`tuple`,  这个参数感觉叫做 scales 更合适一点。用来表示 anchor 对应于原始图片(输入图片) 的 scale。
* `ratios` : `tuple` 表示 anchor 的 长宽 比的一个参数
* `clip` : 因为原图的大小是 [0,1] 之间的（归一化后的），产生的 anchor 可能会超出这个区间，所以这个参数是将超出区间的值 clip 一下。
* `steps`： 
* `offsets` ： anchor 中心点的 偏移。`(y,x)`
* `out` : 用来接收返回值的 `NDArray`。



## MultiBoxTarget

用来 给 `anchor` 来生成 **对应** 需要预测的 `target`。

`target`包含：

* `anchor` 对应的坐标偏移
* `anchor` 对应的 `label`

**使用 这个 API 需要注意的一点是：**

* 在 这个 API 中，返回的 `label`：  0 类别是留给 `background` 用的。输入 `label`  是 `0` 的 `bbox` 会变成  `label` `1`。
* 在 使用的时候，会用 `MultiBoxDetection`， 这个在处理的时候，会将 `label`  减一 , 这样就和 `MultiBoxTarget` 里面的操作抵消了。



`mxnet.ndarray.contrib.MultiBoxTarget(anchor=None, label=None, cls_pred=None, overlap_threshold=_Null, ignore_label=_Null, negative_mining_ratio=_Null, negative_mining_thresh=_Null, minimum_negative_samples=_Null, variances=_Null, out=None, name=None, **kwargs)`

* `anchor` : 就是用 `MultiBoxPrior` 生成出来的 `anchor`.
  * 需要 **reshape 一下**  ，
  * 一个 `3-D` ndarray, `(1, height*width*num_anchors_per_cell, 4)`。
  * `anchor`的信息是需要用来计算 `target` 的。
* `label` :  `是一个 3-D ndarray` . `[bs, num_gt, 5]` : `5` 代表 `[gt_label, xmin, ymin, xmax, ymax]` , `num_gt`  表示这张图中 `gt_bbox` 的数量。注意，这个参数是用来 传入 `gt` 的。
  * 这里的 `gt_label` 还是不用考虑 `background` 的。
  * 当然，一个 `batch` 内，每个样本的 的 `num_gt` 是不一样的， 需要填充成 一样的，怎么填充呢？`[-1., -1., -1., -1., -1.]` 用 `-1` 来填充。
* `cls_pred` : 神经网络对 `anchor` 对应类型的预测输出   `3-D` ndarray,
  *  `[bs, num_class+1, num_anchors]`。
  *  这个参数 的目的是获取 **`bbox`**  类别的个数。`+1` 是包括了 `background`。
* `overlap_threshold` (默认 .5):  iou overlap 超过此值才被看作 positive 样本。 
* `ignore_label` : 如果一个 `anchor` 被 `ignore` 了，这个 `anchor` 的标签就是 这个 `ignore_label`.
  * `ignore` : positive samples ， negative samples，ignored samples。
* `negative_mining_ratio`: 默认值是 -1, 不开启 negative mining。这个参数表示 Max negative to positive samples ratio。如果是 3, 那么采的 negative samples 的数量是 positive samples 的 3 倍。
* `negative_mining_thresh` ：默认 0.5, 低于这个 thresh，才看作 negative 样本。 
*  



**返回值**

* 是一个 list，里面有三个元素
  * `anchor` 与其所预测的 的 bbox 的偏移 。`shape`  是 `[bs, num_anchors*4]` ? 是 x,y 坐标，还是 `center h，w` 坐标。经过看源码，是 中心点加  `h,w` 坐标。
  * `mask` ，用来遮掩 **不需要的负例 anchor** 。`shape` 是 `[bs, num_anchors*4]` . 要遮挡的部分设置成 0 就 ok 了。
  * `anchor`  的   `gt_label` ， `shape` 是 `[bs, num_anchors]`，
    *  这个值 `0` 代表 `background`，`ignore_lable` 表示 `ignored` 的 `anchor`。`>0` 的代表 前景 类别。


## MultiBoxDetection

这里面会执行 `NMS`。

**关于返回值**

* `[bs, num_anchors, 6]` , 其中 6 代表 `[class_id, confidence, xmin, ymin, xmax, ymax]` , `class_id=-1` 代表背景。 `[xmin, ymin, xmax, ymax]` 这些是 **归一化的**，需要乘原图大小来还原成原图的 bbox。



`mxnet.ndarray.contrib.MultiBoxDetection(cls_prob=None, loc_pred=None, anchor=None, clip=_Null, threshold=_Null, background_id=_Null, nms_threshold=_Null, force_suppress=_Null, variances=_Null, nms_topk=_Null, out=None, name=None, **kwargs)`

* `cls_prob` : 网络预测的类别输出，shape 为`[bs, num_class+1, num_anchors]` , `+1` 是添加了 `background` 类别，作为 `label 0`.
  * 注意，这里是 `prob`， 不是 `logits`
* `loc_pred` : 预测出来的 `anchor` 偏移： `[bs, num_anchors*4]`
* `anchor` :  `MultiBoxPrior` 的返回值 `[1, num_anchors, 4]` 
* `clip` :  `clip` 出界的 bbox。
* `threshold` :  `cls_prob` 的 `threshold`， 低于这个 `threshold` 的是不会输出的
* `background_id` : 用来指定 `backgound_id` ，默认是 `0`。
* `nms_threshold` : `nms` 的 `iou threshold`
* `force_suppress` : 
* `variances` :  和 `target` 的对应上就可以了吧。
* `num_topk` : 
* `out` :

