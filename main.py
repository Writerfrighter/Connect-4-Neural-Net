import network
import numpy as np
import random
from copy import deepcopy

#Game Variables
boardxy = [7,6]
connectL = 4
vectors = [(0,1),(0,-1),(-1,0),(1,0),(1,1),(-1,-1),(-1,1),(1,-1)]
P1Char = "X"
P1Float = 1
P2Char = "O"
P2Float = 0.5

#Network Variables
area = boardxy[0]*boardxy[1]
#XXXXXX Weights and Biases
net = [area*2,50,50,50,boardxy[0]]
generations = 50
instances = 150
iterations = 50
mutationChance = 0.1
mutationAmount = 0.5

mstrNet = network.Network(net)

class Game():
  def __init__(self):
    self.board = []
    self.cols = []
    self.PlayerChar = "X"
    self.win = (False, 0)
    for i in range(boardxy[0]):
      self.board.append([])
      for j in range(boardxy[1]):
        self.board[i].append(" ")
    for i in range(boardxy[0]):
      self.cols.append(0)
  def addPeice(self,x):
    #Add peice to the board at collumn x
    if self.cols[x-1] > boardxy[1] - 1:
      return
    for i in range(boardxy[1]):
      if self.board[x-1][i] == " ":
        self.cols[x-1] += 1
        if self.PlayerChar == P1Char:
          self.board[x-1][i] = P1Char
          self.printBoard()
          if self.winCheck((x-1,i)):
            self.win = (True,P1Char)
            print(self.win)
          self.PlayerChar = P2Char
        else:
          self.board[x-1][i] = P2Char
          self.printBoard()
          if self.winCheck((x-1,i)):
            self.win = (True,P2Char)
            print(self.win)
          self.PlayerChar = P1Char
        break
  def winCheck(self,cord):
    #Checks if surrounding peices are the same, if so runs winHelp
    for i in range(len(vectors)):
      try:
        if self.board[cord[0]+vectors[i][0]][cord[1]+vectors[i][1]] == self.PlayerChar:
          if cord[0]+vectors[i][0] >= 0 and cord[1]+vectors[i][1] >= 0:
            if self.winHelp(vectors[i],cord):
              return True
            else:
              pass
      except IndexError:
        pass
    return False
        
  def winHelp(self,vector,cord):
    #Checks vector for the lenght of connectL to see if they are all the same.
    for i in range(connectL):
      if self.board[cord[0]+vector[0]*i][cord[1]+vector[1]*i] != self.PlayerChar:
        return False
    return True
  def printBoard(self):
    for i in range(boardxy[1]):
      row = "|"
      for j in range(boardxy[0]):
        row += " " + str(self.board[j][boardxy[1]-i-1]) + " |"
      print(row)
    print("")

  def exportBoard(self):
    #Converts board into a 2D list containing 1 & 0 with 2 indexes for each grid on the board for Piece 1 and Piece 2.
    convertedBoard = []
    for i in range(boardxy[1]):
      for j in range(boardxy[0]):
        if self.board[j][i] == "X":
          convertedBoard.append([1])
          convertedBoard.append([0])
        elif self.board[j][i] == "O":
          convertedBoard.append([0])
          convertedBoard.append([1])
        else:
          convertedBoard.append([0])
          convertedBoard.append([0])
    convertedBoard = np.reshape(convertedBoard,(area*2,1))
    return convertedBoard


def runGeneration():
  global mstrNet
  currentGen = []
  results = []
  #Initialize results array
  for i in range(instances):
    results.append([0,0])
  #Starts x games agianst privious best net
  for i in range(instances):
    currentGen.append([Game(),(mstrNet,deepcopy(mstrNet))])
    currentGen[i][1][1].mutate(mutationChance, mutationAmount)
  #Runs games
  for j in range(iterations):
    for i in range(instances):
      firstPlayer = random.randint(0,1)
      peices = 0
      while(currentGen[i][0].win[0] == False):
        peices+=1
        if peices > area:
          #print("Timed Out")
          break
        currentGen[i][0].addPeice(currentGen[i][1][firstPlayer].evaluate(currentGen[i][0].exportBoard())[0])
        if currentGen[i][0].win[0] == False:
          peices +=1
          if peices > area:
            #print("Timed Out")
            break
          currentGen[i][0].addPeice(currentGen[i][1][abs(firstPlayer-1)].evaluate(currentGen[i][0].exportBoard())[0])
      #If win is "X"
      if currentGen[i][0].win[0] and currentGen[i][0].win[1] == "X":
        results[i][firstPlayer] += 1
      #If win is "O"
      elif currentGen[i][0].win[0] and currentGen[i][0].win[1] == "O":
        results[i][abs(firstPlayer-1)] += 1
    #print(results)
  #Return each nets win rate
  winRates = [0]
  #Wins
  for x in results:
    winRates[0] += x[0]
    winRates.append(x[1])
  #Mst win rate init
  winRates[0] = winRates[0]/instances
  #Finds percentage
  for i in range(len(winRates)):
    winRates[i] = winRates[i]/iterations
  print(winRates)
  print(np.argmax(winRates))
  if (np.argmax(winRates) != 0):
    mstrNet = deepcopy(currentGen[np.argmax(winRates)-1][1][1])
  
      
def main():
  for i in range(generations):
    runGeneration()
  #game = Game()
  #net1 = network.Network(net)
  #net2 = network.Network(net)
  #for i in range(8):
    #game.addPeice(net1.evaluate(game.exportBoard())[0])
    #game.addPeice(net2.evaluate(game.exportBoard())[0])
main()