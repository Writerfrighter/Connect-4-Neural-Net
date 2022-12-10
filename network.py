import random
import numpy as np
pn = [-1,1]
class Network(object):
  def __init__(self,sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  #Returns activations of the output neurons
  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a)+b)
    return a

  #Return index of highest activation
  def evaluate(self, data):
    return([(np.argmax(self.feedforward(data))+1)])

  #"mc" (mutationChance) chance to offset weights and biases by "ma" (mutationAmount)
  def mutate(self,mc,ma):
    #Mutate Weights
    for w in range(len(self.weights)):
      for x in range(len(self.weights[w])):
        for i in range(len(self.weights[w][x])):
          if random.random() < mc:
            self.weights[w][x][i] = self.weights[w][x][i]+(random.random()/(1/ma))*random.choice(pn)
    #Mutate Biases
    for b in range(len(self.biases)):
      for j in range(len(self.biases[b])):
        if random.random() < mc:
          self.biases[b][j][0] = self.biases[b][j][0] + (random.random()*ma*random.choice(pn))

#Sigmoid :O
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))