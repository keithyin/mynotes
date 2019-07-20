# 文本挖掘



## 第一周

**如何表示文本数据**

* string of **characters** : `a dog is chasing a boy on the playground`
* sequence of **words**: `[a, dog, is , chasing, a, boy, on, the, playground]`
* sequence of words + pos tags : `[Det, Nonu, Aux, Verb, ...]` 
* `+ syntactic structures` : 
* `+ Entities and relations` :
* `+ Logic Predicates`: 
* `+ Speech Acts` : 

----

**Word Association mining and Analysis**

* `word association` 是什么？
  * `paradigmatic relation` : 如果 `A&B` 存在 `paradigmatic relation` , 那么 他们俩可以相互替换。比如：`cat & dog`, `Monday & Tuesday`.  
  * `syntagmatic relation` : 如果 `A&B` 存在 `syntagmatic relation` ，那么他们俩可以组合起来。例如：`cat & sit`, `car & drive`
  * `A & B` 不仅可以是 `word`， 也可以是 `phrase`。

----

**如何挖掘 word association**

直觉：如果两个 `word` 的**上下文相似**，那么他们有很大概率是 `paradigmatic relation`

* `My cat eats fish on Saturday`
* `My dog eats meat on Saturday`

直觉：一起经常出现的

- `My cat eats fish on Saturday`
- `My dog eats meat on Saturday`

----

**Paradigmatic Relation Discovery**

* `Word Context` as  `Pseudo Document`
* 上下文的相似性



## 第二周

**Syntagmatic Relation Discovery ： Correlated Occurrences**

* 使用条件熵来建模

----

**划重点**

$H(X_{w1}|X_{w2})$ 和 $H(X_{w1}|X_{w3})$ 是可以比较的，因为他们有相同的上界，$H(X_{w1})$ 。但是 $H(X_{w1}|X_{w2})$ 和 $H(X_{w3}|X_{w2})$ 是不可以比较的，因为他们的上界不同。

----

**Topic Mining : discover content in the text**

* Topic 是事先确定的吗？ 还是可以自由聚类
* 第一步： discover k topics
* 第二步：figure out which documents cover which topics

Topic Mining 定义：

* **input** : N text documents， Number of topics： k
* **output** : k topics $\{\theta_1, ..., \theta_k\}$ , Coverage of topics in each documents. 

**如何确定 $\theta_j$** ???????????????????

----

**如何确定 Topic**

* Topic == Term 
  * `sports, travel, ..., science`
  * Parse text in Corpus to obtain candidate terms (term==word)
  * Design a scoring function to measure how good each term is as a topic
    * 喜欢具有代表性的 `term` （高频）
    * 避免太高频的 （`the` 这样的）
    * 用下 `TF-IDF`
    * 用一下领域专业知识。
  * Pick K terms with the highest scores but try to minimize redundancy
  * 这种方法，问题多多
* Topic == **Word Distribution** , Vocabulary set ...
  * 输入： `Corpus, k, Vocab`
  * 输出 ：$\theta_j$ 和 document 的 topic coverage



**如何计算 coverage of topics in each document**

* 统计 `terms` 在文章中的出现频数，然后 `normalize` 一下
  * 这种方法，问题多多
* ​

----

**使用生成模型**

* 生成模型假设的是如何生成数据的

$$
\text{P}(Data|Model, \Lambda) 
$$

----

**统计语言模型**

* can regarded as a probabilistic mechanism for "generating" text - thus also called a "generative" model
* Unigram LM : 简单的统计语言模型，假设每个 word 的生成过程是独立的。$p(w|\theta)$ 
  * 不同话题的 document 是从不同的语言模型中生成的。

----

**参数估计方法**

* Maximum Likelihood 和 Bayesian（多一个参数的先验分布）
* ​



## 第三周

**How can we get rid of these common words ?**

* 使用两个 word distribution 生成文档， **Background Topic**
* 由于 word 是由两个分布合作生成的，如果 **Background Topic** 中某个 word 的概率比较高，那么在 Topic Model 中的概率就比较低。

----

**Probabilistic Latent Semantic Analysis (PLSA)**

* document as a sample of mixed Topics
* ​

----

**疑问？？？？？？？？？？？？？？？？**

* 为什么说 PLSA 不是生成模型
* 生成模型，判别模型，监督学习，半监督学习，无监督学习
* EM算法和VI算法的区别

## 第四周





## Similarity Measure

**Expected Overlap of Words in Context (EOWC)**

* ​

**TF-IDF**

* TF：
* IDF：



## 思考

* 重新思考一下 **word2vec** 算法，`CBOW` 和 `SKIP-gram`，一个是上下文相似度，一个根据当前词搞出上下文。
* 生成模型的两个功能：
  * 输出概率（else）
  * 生成样本（GAN）
* EM 算法与 变分推断：EM来求是似然函数或者后验概率最大时对应的参数，变分推断是用来求参数后验分布！！！



## Glossary

* `paradigmatically related` : 词行变化的相关
* `syntagmatically related` ：  语法上的相关
* `lexical analysis` : 也叫 `part-of-speech tagging`（词性标注）, 名词，动词，介词标注。
* `syntactic analysis (parsing)` : `parsing tree`，标注句子的结构，名词短语，介词短语。。 可以看出，比 `lexical analysis` 的抽象层级更高。
* `semantic analysis` : 文本变符号！！
  * 命名实体识别
  * 词意消歧
  * 情感分析
* `Pragmatic Analysis（speech act）` :  理解具体所表达的目的。
* `topic` : main idea discussed in text data