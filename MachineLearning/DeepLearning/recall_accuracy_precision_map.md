# 几个标准总结



## 分类

### Accuracy（准确率）

**给定测试数据集，分类器正确分类的样本与总样本之比。**
$$
accuracy = \frac {right}{all}
$$


## 信息检索

**情景描述：给定 query，从 document 中找到相关的 文件**

### TP，FP，FN，TN

* TP ： true positive，正确的被判定为正类
* FP： false positive，错误的被判定为正类
* TN：true negative，负类被判定为负类。（正确的负类）
* FN：false negative，正类被 判定为 负类。（错误的 负类）

### Precision

$$
precision = \frac{TP}{TP+FP}
$$

正确被检索的 items 占 总共检索的 items 的比例。

**例子：**

假设，输入一条 query，返回了 N 个 items，其中 M 个 items 是正确的，那么 
$$
precision = \frac {M}{N}
$$


### recall

$$
recall = \frac {TP}{TP+FN}
$$

正确被检索的 items 占 所有正确 items 的比例。 $TP+FN$ 代表所有的 正例样本。



### MAP

[http://blog.sina.com.cn/s/blog_662234020100pozd.html](http://blog.sina.com.cn/s/blog_662234020100pozd.html)







## 总结

precision就是**找得对**，召回率就是**找得全**。



## 参考资料

[http://www.cnblogs.com/sddai/p/5696870.html](http://www.cnblogs.com/sddai/p/5696870.html)