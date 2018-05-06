# tensorflow中使用clip
**在训练深度神经网络的时候,我们经常会碰到`梯度消失`和`梯度爆炸`问题,scientists提出了很多方法来解决这些问题,本篇就介绍一下如何在tensorflow中使用clip来address这些问题**
```python
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
```
在调用`minimize`方法的时候,底层实际干了两件事:
- 计算所有 `trainnable variables` 梯度
- apply them to `variables`

随后, 在我们 `sess.run(train_op)` 的时候, 会对 `variables` 进行更新

## clip
那我们如果想处理一下计算完的 `gradients` ,那该怎么办呢?
官方给出了以下步骤
1. Compute the gradients with compute_gradients(). 计算梯度
2. Process the gradients as you wish.              处理梯度
3. Apply the processed gradients with apply_gradients(). apply处理后的梯度给`variables`

这样,我们以后在`train`的时候就会使用 processed gradient去更新 `variable`
**框架:**
```python
# Create an optimizer.
opt = GradientDescentOptimizer(learning_rate=0.1)

# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(loss, <list of variables>)

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

# Ask the optimizer to apply the capped gradients.
opt.apply_gradients(capped_grads_and_vars)
```
**例子:**
```python
#get all trainable variables in your model
params = tf.trainable_variables()

#create an optimizer
opt = tf.train.GradientDescentOptimizer(self.learning_rate)

#compute gradients for params
gradients = tf.gradients(self.losses[b], params)

#process gradients
clipped_gradients, norm = tf.clip_by_global_norm(gradients,max_gradient_norm)

train_op = opt.apply_gradients(zip(clipped_gradients, params),global_step=self.global_step))
```
这时, `sess.run(train_op)` 就可以进行训练了
