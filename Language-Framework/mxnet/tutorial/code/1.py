import mxnet as mx

a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = mx.sym.Variable('c')
d = 2 * a + b + c

e = d.simple_bind(mx.cpu(), b=(1,))

y = e.forward(is_train=True)
e.backward(out_grads=mx.nd.array([1]))


for grad in e.grad_arrays:
    print(grad.asnumpy())
for val, grad in zip(e.arg_arrays, e.grad_arrays):
    val -= grad
e.forward(is_train=True)

print("first forward", e.outputs[0].asnumpy())
# e.backward(out_grads=mx.nd.array([2.]))

# for value in e.arg_arrays:
#     print(value.asnumpy())
#     value += mx.nd.array([1])
#
# for value in e.arg_arrays:
#     print(value, value.asnumpy())
#
# for i in range(10):
#     e.forward()
#     for value in e.outputs:
#         print(value.asnumpy())

# ar1 = mx.nd.array([1, 3])
# ar2 = mx.nd.array([1, 2])
#
# print(id(ar1))
# ar1 -= ar2  # point to the same thing
# print(id(ar1))

