from copy import deepcopy
import random
import numpy as np

file = open("data.txt","r")

#Board configurations converted to neural net format
boards = []
#Move decided by the Mini-Max alg found here https://replit.com/@Writerfrighter/Connect-Four-Database-Maker
results = []

#Training, validation, and testing data
tr_d, va_d, te_d = [[],[]],[[],[]],[[],[]]

def loadData(tr_dCount,va_dCount,te_dCount):
  transformData()
  transformDataTwo(tr_dCount,va_dCount,te_dCount)
  training_inputs = [np.reshape(x, (84, 1)) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = list(zip(training_inputs, training_results))
  validation_inputs = [np.reshape(x, (84, 1)) for x in va_d[0]]
  validation_data = list(zip(validation_inputs, va_d[1]))
  test_inputs = [np.reshape(x, (84, 1)) for x in te_d[0]]
  test_data = list(zip(test_inputs, te_d[1]))
  return (training_data, validation_data, test_data)
  
def transformDataTwo(tr_dCount,va_dCount,te_dCount):
  bd = deepcopy(boards)
  rslts = deepcopy(results)
  #Splits data into tr_d, va_d, and te_d
  for i in range(tr_dCount):
    j = random.randint(0,len(bd)-1)
    tr_d[0].append(bd[j])
    tr_d[1].append(rslts[j])
    bd.pop(j)
    rslts.pop(j)
  for i in range(va_dCount):
    j= random.randint(0,len(bd)-1)
    va_d[0].append(bd[j])
    va_d[1].append(results[j])
    bd.pop(j)
    rslts.pop(j)
  for i in range(te_dCount):
    j = random.randint(0,len(bd)-1)
    te_d[0].append(bd[j])
    te_d[1].append(results[j])
    bd.pop(j)
    rslts.pop(j)
def transformData():
  #Converts data into neural net format
  t = 0
  for line in file:
    board = line.split(",")
    if board[42][0] != "-":
      boards.append([])
      for i in range(42):
        if board[i] == "1":
          boards[t].append([1])
          boards[t].append([0])
        elif board[i] == "2":
          boards[t].append([0])
          boards[t].append([1])
        else:
          boards[t].append([0])
          boards[t].append([0])
      results.append(int(board[42][0]))
      t+=1

def vectorized_result(j):
    """Return a 7-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...6) into a corresponding desired output from the neural
    network."""
    e = np.zeros((7, 1))
    e[j] = 1.0
    return e
  

    