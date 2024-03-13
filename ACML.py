# PERCEPTRON TRAINING ALGORITHM
import random


#testing data
D = {((3, 1), 1),
     ((2, 2.5), 0), 
      ((2, 1.5), 1), 
      ((4, 3), 1), 
      ((3, 3), 0)},

# Variables
weights = [2,3.5,6]
learning_rate = 0.01
threshold = -1


array_of_inputs = []
targets =  []

for i in D:
    print(f"i: {i}")
