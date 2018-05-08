# Attention 与 CTC 对比

**学到一种对齐方式 vs 压根就不对齐(通过 merge 解决问题)**



* 为什么说 CTC 不对齐呢？
  * 和目标函数相关，目标是最大化 所有合理的 alignment 的概率和。
  * alignment 合理，最大化概率和就可以了，**不管它对齐是否正确**



**attention decoder**

* 每个 step，根据当前的隐层状态进行 query，确定要注意的 encoder state
* 然后再进行解码
* **decoder** 的输入是和 **encoder state** 对齐的结果。



**ctc 的时候（最大化 合理的 alignment 的概率和，合理：可以 merge 成 ground-truth 的 alignment）**

* 通过动态规划，该 merge 的就 merge 。
* 最大化 可以 merge 成 target 的所有 串的概率
* ctc 的输入是 严格 和 encoder 的输入时刻相关的，**merge** 可以看作一种后处理方式





**相同点**

* ground truth label 的给的方式是一样的。



**不同点**

* 对齐(采用 attention 的方式进行对齐)，与不对齐



## ctc 性质

* 条件独立性假设：给定输入 x ，输出是条件独立的，
  * 完全靠 声学模型解码，不靠语言模型解码。好处是：声学模型和语言模型独立。声学模型可以迁移到其它领域。坏处：没有语言模型，准确率可能不好。
  * 当然也可以额外加入 语言模型解码。

$$
P(y|x) = P(y_0, y_1, ..., y_s|x) = P(y_0|x)P(y_1|x)...
$$

* CTC only allows **monotonic alignments**. （考虑单调递增函数，X 轴是输入，Y轴是输出。） In problems such as speech recognition this may be a valid assumption. For other problems like machine translation where a **future word in a target sentence can align to an earlier part of the source sentence, this assumption is a deal-breaker.**
* Another important property of CTC alignments is that they are *many-to-one*. Multiple inputs can align to at most one output.
* The many-to-one property implies that the **output can’t have more time-steps than the input**.  



----

## Attention

* 有语言模型，有对齐。



## 对比

**不同点**

* attention 有语言模型， CTC 无语言模型
* attention 学习对齐方式，CTC 不关心对不对齐
* many-to-one 的方式不一样，attention 是特征 多个融合，解码一个 time-step，CTC 是解码结果 merge。
  * attention 对齐时，$x_0, x_1, .., x_T$ 在每个解码 `step` 可以复用，ctc merge时，用一次就没了。
* CTC 是一种 monotonic 对齐，而 attention 无所谓。
* ​

**相同点**

* 都是 many-to-one 方式。
* ​



## 参考资料

[https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)