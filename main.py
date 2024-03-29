import network
import numpy as np
import random
import dataLoader


#Game Variables
boardxy = [7,6]
connectL = 4
vectors = [(0,1),(1,0),(1,1),(1,-1)]
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
      
  def addPiece(self,x):
    
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
          if self.winCheck():
            self.win = (True,P1Char)
          self.PlayerChar = P2Char
          return
        else:
          self.logMove(x)
          self.board[x-1][i] = P2Char
          self.printBoard()
          #Win check for P2
          if self.winCheck():
            self.win = (True,P2Char)
          self.PlayerChar = P1Char
          return
        break
    if self.PlayerChar == P1Char:
      self.PlayerChar = P2Char
    else:
      self.PlayerCahr = P1Char
    print("Invalid col")
  def winCheck(self):
    #Checks if surrounding peices are the same, if so runs winHelp
    #Loops through all locations on the board
    for i in range(boardxy[0]):
      for j in range(boardxy[1]):
        #For each direction, check the current location and the piece next to in the vector is the current peice
        for o in range(len(vectors)):
          try:
            #Piece next to is the same
            if self.board[i+vectors[o][0]][j+vectors[o][1]] == self.PlayerChar and self.board[i][j] == self.PlayerChar:
              #Make sure we are not iterating backward through the list
              if i+vectors[o][0] >= 0 and j+vectors[o][1] >= 0:
                if self.winHelp(vectors[o],(i,j)):
                  return True
                else:
                  pass
          except IndexError:
            #In the event that the vector is refering to a peice above the top row
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
        row += " " + str(self.board[j][-i-1]) + " |"
      print(row)
    print("")

  #Network Game Functions
    
  def exportBoard(self):
    #Converts board into a 2D list containing 1 & 0 with 2 indexes for each grid on the board for Piece 1 and Piece 2.
    #Fills board by cols, left to right
    #6 . . . . . . .
    #5 . . . . . . .
    #4 . . . . . . .
    #2 . . . . . . .
    #1 . . . . . . .
    #  a b c d e f g
    #So a1, a2, a3, and so on
    convertedBoard = []
    for i in range(boardxy[0]):
      for j in range(boardxy[1]):
        if self.board[i][j] == "X":
          convertedBoard.append([1])
          convertedBoard.append([0])
        elif self.board[i][j] == "O":
          convertedBoard.append([0])
          convertedBoard.append([1])
        else:
          convertedBoard.append([0])
          convertedBoard.append([0])
    convertedBoard = np.reshape(convertedBoard,(area*2,1))
    return convertedBoard

  def logMove(self,move):
    #Log move if human player to a game log to train the network.
    if self.PlayerChar == self.humanPlayer or self.humanPlayer == "Both":
      self.log.append((self.exportBoard(),move-1))

def getCol(game):
  #Prevent invalid arguments for column selection
  try:
    game.addPiece(int(input("Which column would you like to add a " + game.PlayerChar + " to? ")))
  except:
    print("Invalid")
    getCol(game)
    
def runPlayerGame():
  #Run a game between 2 People
  game = Game()
  game.humanPlayer = "Both"
  while game.win[0] == False:
    getCol(game)
    if game.win[0] == False:
      getCol(game)
  print("Training Network")
  net.update_Network(game.log,100)
  print("Done")

      
def runGame():
  #Run a game with the Feedforward network and a player
  game = Game()
  while game.win[0] == False:
    if game.humanPlayer == "X":
      getCol(game)
    else:
      game.addPiece(net.findMove(game.exportBoard())[0])
    if game.win[0] == False:
      if game.humanPlayer == "O":
        getCol(game)
      else:
        game.addPiece(net.findMove(game.exportBoard())[0])
  if game.win[1] == game.humanPlayer:
    print("Congrats!")
    print("Updating Network...")
    net.update_Network(game.log,100)
    print("Done")
  else:
    print("You lost :(")
  print("Starting new game...")
  
def main():
  training_data, validation_data, test_data = \
  dataLoader.loadData(30000,1000,1000)
  net = network.Network(netLayers)
  net.SGD(training_data,100,1000,0.2,test_data=test_data)
  if input("Write to file? ") == "Yes": net.writeToFile()
main()