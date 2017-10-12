# Hierarchical Learning

## Deep Successor Reinforcement Learning

> Learning robust value functions given raw observations and rewards is now possible with model-free and model-based deep reinforcement learning algorithms. There is a third alternative, called Successor Representations(SR).

* model-free
* model-based
* successor representation



> Successor Representation which decomposes the value function into two components:
>
> * a reward predictor
> * a successor map
>   * represent the **expected future state occupancy from any given state** and the **reward predictor maps states to scalar rewards**. 
>
> The value function can be computed as the **inner product between the successor map** and the **reward weights**.

这里需要注意的是：是 reward predictor， 这个模型的意思是打算搞 reward 了



> The value function at a state can be expressed as the dot product between the **vector of expected discounted future state occupancies** and the **immediate reward** in each of those successor states.



**Introduction**

SR's appealing properties:

* combine computational efficiency comparable to model-free algorithms with some of the flexibility of model-based algorithms   ??????
* SR can adapt quickly to changes in distal reward, unlike model-free algorithms ?????



## Questions

* ​



## Glossary

* state occupancy
* ​