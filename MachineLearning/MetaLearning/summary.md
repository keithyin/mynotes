Meta-learning, or learning to learn, is the science of systematically observing how diﬀerent machine learning approaches perform on a wide range of learning tasks, and then learning from this experience, or meta-data, to learn new tasks much faster than otherwise possible. Not only does this dramatically speed up and improve the design of machine learning pipelines or neural architectures, it also allows us to replace hand-engineered algorithms with novel approaches learned in a data-driven way.


要解决什么问题
* zero-shot learning
    * Zero-Shot learning method aims to solve a task without receiving any example of that task at training phase. The task of recognizing an object from a given image where there weren’t any example images of that object during training phase can be considered as an example of Zero-Shot Learning task. Actually, it simply allows us to recognize objects we have not seen before.
    * https://medium.com/@cetinsamet/zero-shot-learning-53080995d45f
* one-shot learning
    * Shot Learning refers to Deep Learning problems where the model is given only one instance for training data and has to learn to re-identify that instance in the testing data
    * The solution to this is to train the network to learn a distance function between images rather than explicitly classifying them. This is the central idea behind One-Shot Learning.
* few-shot learning
    * few-shot learning refers to the practice of feeding a learning model with a very small amount of training data, contrary to the normal practice of using a large amount of data.
    * https://medium.com/quick-code/understanding-few-shot-learning-in-machine-learning-bede251a0f67




Meta-Learning
* 参考资料
    * https://zhuanlan.zhihu.com/p/28639662
    * https://medium.com/huggingface/from-zero-to-research-an-introduction-to-meta-learning-8e16e677f78a
* 目的： humans leverage past experiences to learn very quickly a new task from only a handset of examples


学习optimizer：
* What I will call the model (M) which is our previous neural net. It can now be seen as a low-level network. It is sometimes called an optimizee or a learner. The weights of the model are the ■ on the drawings.
* The optimizer (O) or meta-learner is a higher-level model which is updating the weights of the lower-level network (the model). The weights of the optimizer are the ★ on the drawings.


两个任务
* during the meta-forward pass: we use our model to compute gradients(from the loss) that are feed as inputs to the optimizer to update the model parameters, and （就是计算梯度，然后给model更新模型参数）
* during the meta-backward pass: we use our model as a path for back propagating the gradients of the optimizer’s parameters (computed from the meta-loss)


## Meta-Learning with Memory-Augmented Neural Networks
* 目标是解决 one-shot learning 问题
* Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information
* memory-augmented neural network to rapidly assimilate new data, and leverage this data to make accurate predictions after only a few samples.
    * 这样的话，是不是一旦一个新的数据出现，如果想要再判断出它的话，另一个类似的新数据就需要尽快出现。
    * 如果是这样的话，是不是这模型可以用来做 频控？
* 本文中，一个 task 也叫一个 episode。训练的时候，模型的输入序列是 (x1, null), (x2, y1), . . . , (xT , yT−1)。模型的学习目标是： it must learn to hold data samples in memory until the appropriate labels are presented at the next time-step, after which sample-class information can be bound and stored for later user. 
    * Memory中能存多少样本？
* Thus, for a given episode, ideal performance involves a random guess for the first presentation of a class (since the appropriate label can not be inferred from previous episodes, due to label shuffling), and the use of memory to achieve perfect accuracy thereafter.
* 这个模型meta-learn 的目标是：learn to bind data representations to their appropriate labels regardless of the actual content of the data representation or label, and would employ a general scheme to map these bound representations to appropriate classes or function values for prediction.
* an image would appear, and they must choose an appropriate digit label from the integers 1 through 5. Next, the image was presented and they were to make an un-timed prediction as to its class label. The image then disappeared, and they were given visual feedback as to their correctness, along with the correct label. The correct label was presented regardless of the accuracy of their prediction, allowing them to further reinforce correct decisions. After a short delay of two seconds, a new image appeared and they repeated the prediction process. The participants were not permitted to view previous images, or to use a scratch pad for externalization of memory
    * 出来一张图片，用户猜一个数，然后给他们正确的标签。过了几秒，再让用户看图猜标签
    * 这任务不就是考研记忆力的吗。。




## Learning to Learn by Gradient Descent by Gradient Descent
* 学习优化算法
* 做法
    * 传统的训练过程看做一个序列，使用下一步的 loss 辅助更新 优化器的参数，意思就是：上次优化器应该这么优化，会使得当前步的loss变小。学习完毕的优化器在优化当前步的时候，不仅会考虑当前步的loss，也会考虑之后步的loss
    * 有点强化学习的感觉
    
## Matching Networks for One Shot Learning
* 思想是从 support set 中找相似

## Optimization As a Model for Few Shot Learning
* 传统的方法
  * 传统的优化器算法并不是为了更少的迭代次数设计
  * 训练模型的时候，每次都从随机参数开始更新，速度比较慢
  * 当两个任务差距比较大的时候，使用Transfer Learning的效果也不是很好
* 本文：学习一个优化器
