import dynet as dy

m = dy.ParameterCollection()
pW = m.add_parameters((8,2))
pV = m.add_parameters((1,8))
pb = m.add_parameters((8))

# renew the computation graph
dy.renew_cg()

# add the parameters to the graph
W = dy.parameter(pW)
V = dy.parameter(pV)
b = dy.parameter(pb)

# create the network
x = dy.vecInput(2) # an input vector of size 2.
output = dy.logistic(V*(dy.tanh((W*x)+b)))
# define the loss with respect to an output y.
y = dy.scalarInput(0) # this will hold the correct answer
loss = dy.binary_log_loss(output, y)

# create training instances
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()


print(b.value())

# train the network
trainer = dy.SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    # total_loss += loss.value()
    loss.backward()
    trainer.update()
    # if (seen_instances > 1 and seen_instances % 100 == 0):
    #     print("average loss is:",total_loss / seen_instances)
print(b.value())