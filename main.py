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
#Network layer variable
netLayers = [area*2,50,50,50,boardxy[0]]
net = network.Network(netLayers)

class Game():
  def __init__(self):
    self.board = []
    self.cols = []
    self.PlayerChar = "X"
    self.win = (False, 0)
    self.humanPlayer = random.choice([P1Char,P2Char])
    self.log = []
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
          self.logMove(x)
          self.board[x-1][i] = P1Char
          self.printBoard()
          #Win check for P1
          if self.winCheck((x-1,i)):
            self.win = (True,P1Char)
          self.PlayerChar = P2Char
        else:
          self.logMove(x)
          self.board[x-1][i] = P2Char
          self.printBoard()
          #Win check for P2
          if self.winCheck((x-1,i)):
            self.win = (True,P2Char)
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
    #Prints board to console
    for i in range(boardxy[1]):
      row = "|"
      for j in range(boardxy[0]):
        row += " " + str(self.board[j][boardxy[1]-i-1]) + " |"
      print(row)
    print("")

  #Network Game Functions
    
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

  def logMove(self,move):
    if self.PlayerChar == self.humanPlayer:
      self.log.append((self.exportBoard(),move-1))
 
def runGame():
  game = Game()
  while game.win[0] == False:
    if game.humanPlayer == "X":
      game.addPeice(int(input("Which column would you like to add a X to? ")))
    else:
      game.addPeice(net.evaluate(game.exportBoard())[0])
    if game.win[0] == False:
      if game.humanPlayer == "O":
        game.addPeice(int(input("Which column would you like to add a O to? ")))
      else:
        game.addPeice(net.evaluate(game.exportBoard())[0])
  if game.win[1] == game.humanPlayer:
    print("Congrats!")
    print("Updating Network...")
    net.update_Network(game.log,10)
    print("Done")
  else:
    print(":O the Network has learned :D")
  print("Starting new game...")
def main():
  while True:
    runGame()
  
  
  
main()