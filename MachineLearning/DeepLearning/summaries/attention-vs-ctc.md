# Attention 与 CTC 对比

**学到一种对齐方式 vs 压根就不对齐(通过 merge 解决问题)**





**attention decoder**

* 每个 step，根据当前的隐层状态进行 query，确定要注意的 encoder state
* 然后再进行解码
* **decoder** 的输入是和 **encoder state** 对齐的结果。



**ctc 的时候**

* 通过动态规划，该 merge 的就 merge 。
* 最大化 可以 merge 成 target 的所有 串的概率
* ctc 的输入是 严格 和 encoder 的输入时刻相关的，**merge** 可以看作一种后处理方式





**相同点**

* ground truth label 的给的方式是一样的。



**不同点**

* 对齐(采用 attention 的方式进行对齐)，与不对齐



**ctc 性质**

* 条件独立性假设：给定输入 x ，输出是条件独立的

$$
P(y|x) = P(y_0, y_1, ..., y_s|x) = P(y_0|x)P(y_1|x)...
$$

* As mentioned before, CTC only allows *monotonic* alignments. In problems such as speech recognition this may be a valid assumption. For other problems like machine translation where a future word in a target sentence can align to an earlier part of the source sentence, this assumption is a deal-breaker.
* Another important property of CTC alignments is that they are *many-to-one*. Multiple inputs can align to at most one output.
* ​

## 参考资料

[https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)

