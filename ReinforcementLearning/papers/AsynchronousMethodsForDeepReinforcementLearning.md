# A3C (Asynchronous Advantage Actor-Critic)



**key words:**

* asynchronous gradient descent
* actor-critic
* model-free



Using **asynchronous gradient descent** for optimization of deep neural network controllers.



**experience replay 的缺点**

* 内存占用大，要存 1百万个 transitions
* 只能用 off-policy 算法



**A3C**

> we asynchronously execute multiple agents in parallel, on multiple instances of the environment.

这种异步的方法 de-correlates the agents' data into a more stationary process. 由于在每个 time-step，并行的 agents 会经历各种各样的 states。

* 可以用很多 on-policy 的算法了
* off-policy 的算法也能用， 美滋滋
* cpu 就开搞，不用依赖 GPU，更加美滋滋



> One drawback of using one-step methods is that obtaining a reward $r$ only directly affects the value of the state action pair ($s,a$) that led to the reward. The value of other state action pairs are affected only indirectly through the updated value $Q(s,a)$. This can make the learning process low.

one-step 更新的方法 一次只更新一个 (s,a) pair，这样更新速度非常慢。



**理解一下 advantage **:
$$
A(s_t,a_t) = Q(s_t, a_t)-V(s_t)
$$
为什么叫 advantage，看式子，可以这么改写 $Q(s_t,a_t)-\mathbb E(a | s_t)$ ，这个就是取  $a_t$ 所占的优势啦。



**如何异步**

* 每个 thread 维护自己的 环境，求自己的梯度（accumulated），
* 然后异步更新 权重（具体是怎么更新，那个线程到步数了就更新？）
* RMSprop 那部分怎么搞

## Glossary

* on-line RL algorithms : 每个 step 都更新