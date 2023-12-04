import matplotlib.pyplot as plt
import random
from driver import Value
from neural_net import Neuron, Layer, MLP

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

ys = [-1, 1, -1, 1]

n = MLP(3, [4,4,1])

for i in range(200):

    #Forward Pass
    preds = [n(x) for x in xs]
    # print(preds)

    #Loss function
    loss = sum((youts - yreal)**2 for yreal, youts in zip(ys, preds))

    #Gradient Zeroing
    for p in n.parameters():
        p.grad = 0.0

    #Backward Pass
    loss.backward()

    #Update weights
    for p in n.parameters():
        p.data += -0.01 * p.grad

    if i%5==0:
        print(i, loss.data)

print(n([2.0, 3.0, -1.0]))
