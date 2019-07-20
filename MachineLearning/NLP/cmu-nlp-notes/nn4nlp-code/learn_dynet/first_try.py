import dynet as dy

# reates a parameter collection and populates it with parameters.
m = dy.Model()
pW = m.add_parameters((8, 2))
pV = m.add_parameters((1, 8))
pb = m.add_parameters((8))

dy.renew_cg()  # new computation graph. not strictly needed here, but good practice.

# associate the parameters with cg Expressions
# creates a computation graph and adds the parameters to it,
# transforming them into Expressions.
# The need to distinguish model parameters from “expressions” will become clearer later.
W = dy.parameter(pW)
V = dy.parameter(pV)
b = dy.parameter(pb)

x = dy.vecInput(2)  # an input vector of size 2. Also an expression.
output = dy.logistic(V * (dy.tanh((W * x) + b)))

y = dy.scalarInput(0)
loss = dy.binary_log_loss(output, y)

print(x.value())



print(b.value())

trainer = dy.SimpleSGDTrainer(m, learning_rate=.001)
for i in range(1000):
    x.set([1, 1])
    y.set(0)
    # loss.value()
    loss.backward()
    trainer.update()
print(x.value())

print(b.value())

print(output.value())
