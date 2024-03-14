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
learningRate:bool = 0.01
threshold:int = -1

# inupts and targets
inputs:list = [i[0] for i in D]
targets:list = [j[1] for j in D]

print(f"INPUTS: \n{inputs}")
print(f"\nTARGETS: \n{targets}")

def perceptronTrainingAlgorithm(tresholdValue: int, learningRateValue:int , weightValues: list, targetValues: list, inputValues:list):
    summation = (inputValues[0]*weightValues[0])+(inputValues[1]*weightValues[1])
    if summation>tresholdValue:
        output = 1
    elif summation < tresholdValue:
        output = 0

    return output       
    
    updatedThreshold = tresholdValue - (learningRateValue*(output-trueValue))


epoch = 0
while epoch < 100:
    for input in inputs:
        print(epoch)
        epoch+=1
     


