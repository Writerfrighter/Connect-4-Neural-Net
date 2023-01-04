import random
import numpy as np

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

  def update_Network(self, game_log, eta):
    random.shuffle(game_log)
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in game_log:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(game_log))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(game_log))*nb for b, nb in zip(self.biases, nabla_b)]

  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation)+b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
      sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)

  def evaluate(self, data):
    return([(np.argmax(self.feedforward(data))+1)])
    
  def writeToFile(self):
    #Clear file
    file = open("Network/Sizes.txt","w")
    file.write("")
    #Write layer sizes to file
    file = open("Network/Sizes.txt","a")
    file.write(str(self.sizes) + "\n")
    #Write all weights to the file
    layer = 0
    for layers in self.weights:
      layer += 1
      file = open("Network/Weights/"+str(layer)+".txt", "w")
      for neurons in layers:
        for weights in neurons:
          file.write(str(weights))
          file.write("\n")
      file.close()
    #Write all baises to the file
    layer = 0
    for layers in self.biases:
      layer += 1
      file = open("Network/Biases/"+str(layer)+".txt","w")
      for neuronBiases in layers:
        file.write(str(neuronBiases[0]))
        file.write("\n")

  def readFile(self):
    file = open("Network/Sizes.txt","r")
    #Convert network size from str to list
    sizes = file.read().split("\n")[0].strip(" ").strip("]").strip("[").split(",")
    #Transform str to int
    for i in range(len(sizes)):
      sizes[i] = int(sizes[i])
    #Write size to network
    self.sizes = sizes
    
    #Reset weights
    self.weights = []
    #Write weights to network
    for l in range(len(self.sizes)-1):
      self.weights.append([])
      file = open("Network/Weights/"+str(l+1)+".txt","r")
      weights = file.read().split("\n")
      for n in range(self.sizes[l+1]):
        self.weights[l].append([])
        for w in range(self.sizes[l]):
          self.weights[l][n].append(weights[n*sizes[l]+w])

    #Reset biases
    self.biases = []
    #Write biases to network
    for l in range(len(self.sizes)-1):
      file = open("Network/Biases"+str(l+1)+".txt","r")
      self.biases.append([])
      for b in range(self.sizes[l+1]):
        self.biases[l].append(file[b])
  def cost_derivative(self, output_activations, y):
    return (output_activations-y)

#Returns a value that is a compression of the real number line into 0 & 1
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Returns the "Senitivity" of a certain weight or bias
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))