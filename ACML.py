# PERCEPTRON TRAINING ALGORITHM
import random


#testing data
D = {((3, 1), 1),
     ((2, 2.5), 0), 
      ((2, 1.5), 1), 
      ((4, 3), 1), 
      ((3, 3), 0)}

# parameters randomly assigned
weights = [2,3.5]
learning_rate = 0.01
threshold = -1
# inupts and targets
inputs = [i[0] for i in D]
targets = [j[1] for j in D]

print(f"INPUTS: \n{inputs}")
print(f"\nTARGETS: \n{targets}")

array_of_inputs = []
targets =  []


