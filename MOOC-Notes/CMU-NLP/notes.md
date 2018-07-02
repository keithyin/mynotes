# 10. Debugging Neural Nets for NLP

**Isolate the problem**



**hyper-parameter**

* network size/depth/structure
* mini-batch creation strategy (hard-mining etc.)
* optimizer/learning rate



**A Typical Situation**

* you've implemented a nice model
* you've looked at the code, and it looks OK.
* You accuracy on the test set is bad !!

**Possible Causes**

* Training time problems
  * Lack of model  capacity
  * inability to train model properly
  * just a bug
* Decoding time bugs
  * Disconnect between training and decoding
  * Failure of search algorithm
* Over-fitting
* Mismatch between optimized function and your metric



**Debugging at Training Time Problems**

* Look at the loss function calculated on the training set
  * is the loss function going down
  * Is ti going down basically to zero if you run training long enough (20-30 epochs)
* if not, you have a training problem. (**You can't over fit the training set**)
* 可能导致的原因：
  * model is too weak。loss 会下降一些，但是不多。



**Be Careful of  Deep Models**

* 使用残差连接



**Learning Rate**

* start from an initial learning rate then gradually decrease
* SGD default=0.1 Adam=0.001



**Initialization**

* result in different sized gradients



**Mini-Batch in RNNs**

* Sorting and Bucketing
* 如果不 sort 的话，会很浪费时间。但是会降低性能，因为相同长度的句子倾向于具有相同的单词。



**Debugging at Decoding Time**

* Your decoding should get the same score as loss calculation
* Test this:
  * Calculate loss of reference
  * Perform **forced decoding**, where you decode, but tell your model the reference work at each time step.
  * The score of these two should be the same
* create a unit test doing this



**Debugging Search**

* As you make search better, the model score should get better (almost all the time)
* Run search with varying beam sizes and make sure you get a better overall model score with larger sizes.
* create a unit test for this



**Quantitative Analysis**

* Measure gains quantitatively. what is the phenomenon you chose to focus on? Is that phenomenon getting better?
  * 如果你 focused on low-frequency words: 那就看一下 is accuracy on low frequency words increasing?
  * 如果你 focused on syntax: 那就要看一下 is syntax or word ordering getting better, are you doing better on long-distance dependencies?
  * 意思就是，focus 哪里，就要对哪部分主要分析。



**Battling Over-Fitting**

* training loss converges well, but test loss diverges
* **Your Neural Net can Memorize you Training Data**



**Mis-match between Optimized Function and Evaluation Metric**

* 优化的目标好 并不意味着 评估的时候效果就好。



**Reminder**

* optimizer
* learning rate
* look at your data



# 15. Structured Prediction 

* 对于 machine translation 来说，有 无限种可能结果，称之为 Structured Prediction



**Many Varieties of Structured Prediction**

* Models

  * RNN based decoders
  * convolution/ self attentional decoders
  * CRF with local factors
  * autoregressive models

* Training algorithms

  * Structured perceptron, structured large margin
  * Sampling corruptions of data
  * Exact enumeration with dynamic programs
  * Reinforcement learning / minimum risk training

  ​

**Structured prediction model**

* Modeling the output structure
* Output 之间是非独立的



**Training Structured Models**

* Simplest training method "teacher forcing"
  * Just feed in the correct previous tag



**Teacher Forcing's Problems**

* Exposure Bias （Local Normalized Model）
  * 因为在训练的时候 feed 的是真实的 label，在测试的时候没法做到
  * The model has only been exposed to correct input in the training, So it has a limit in tthe types of things that has been exposed to. 所以在测试的时候就凉凉了。
* **Pre-train with teacher forcing, then fine-tune with more complicated algorithm**
* Sample Mistakes in Training, 使用 scheduling 方法有计划的 采随机样本。
  * 刚一开始用 Teacher Forcing 然后缓慢的引入 错误样本。
* **Dropout Inputs,  decoder直接输入 0 向量。**
* Corrupt Training Data, 标签可以引入噪声。



**Global Normalized models**

* **The denominator is too big to expand naively**
* Must do something tricky:
  * Consider only subset of hypotheses
  * Design the model so we can efficiently enumerate all hypotheses



# 14. Reinforcement Learning for NLP

* Environment X
* ability to make actions A
* get a delayed reward R



**Why Reinforcement Learning in NLP**

* may have **typical reinforcement learning scenario**
  * a dialog where we can make responses and will get a reward at the end
* may have **latent variables**, 
  * where we decide the latent variable, the get a reward based on the configuration.
* may have a **sequence-level error function** 
  * such as BLEU score that we cannot optimize without first generating a whole sentence.





**Self Training**

* Sample or argmax according to the current model

$$
\hat Y \sim P(Y|X) or\hat Y=\arg\max_Y P(Y|X)
$$

* Use this sample (or samples) to maximize likelihood

$$
loss = -\log P(\hat Y|X)
$$

**Stabilizing Reinforcement Learning**

* like other sampling-based methods, reinforcement learning is unstable
* particularly unstable when **using bigger output spaces**
  * Bigger action spaces
  * Not good at machine translation
  * good at binary classification
* a number of strategies can be used to stabilize
  * **Adding a Baseline**
    * 基本想法： we have expectations about our reward for a particular sentence.
  * **Increasing Batch Size**
  * **Warm Start**
    * Start training with maximum likelihood, then switch over to REINFORCE



**When to Use Reinforcement Learning**

* correct actions are not given
* correct actions are not given but computation structure doesn't change
  * differentiable approximation (Gumbel Softmax) may be more stable
* can train using MLE, but want to use a non-decomposable loss function.
  * may be yes, but may other methods (max margin, min misk) also






# 20. Dialogue

**Type of dialogue**

* task driven
* chat



**Two Paradigms**

* Generation-based Models
  * take input, generate output
  * Good if you want to be creative
* Retrieval-Based Models
  * take input, find most appropriate output
  * Good if you want to be safe



**Dialog 存在的难题**

* **Dialog More Dependent on Global Coherence**
* **Dialog allows much more varied responses**
  * 对于 translation 来说，可以有单词上的差距，但是意思是一样的
  * 但是对于 dialog 来说，意思可能都不一样
* **Diversity is a Problem for Evaluation**
  * Translation uses BLEU score, while imperfect, not horrible
  * 既然 Evaluation 是个大问题，那就学习如何评分吧。Learning to Evaluation
* **Dialog Agents should have Personality**





**Generation-Based Models**

* train machine translation system to perform translation from utterance to response
* lots of filtering, etc., to make sure that the extracted translation rules are reliable
* Unlike machine translation: 上下文信息有时候有用，有时候是无用的。
* ​



**Retrieval-Based Models**

* 很多问题都可以使用模板来回答的，只需要从 corpus 中找到最合适的 response 就好了。



**Task-Driven Dialog**

* **Task-Completion Dialog**
  * Natural language understanding to **fill the slots** in the frame based on the user utterance
  * **Dialog state tracking** to keep track of the overall dialog state over multiple turns.
  * **Dialog control** to decide the next action based on state
  * Natural language generation to generate utterances based on current state
  * ​