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

print(f"\nINPUTS: \n{inputs}")
print(f"\nTARGETS: \n{targets}")




class SimplePerceptron:
    '''
    Expects threshold value, a list weight values, lis of input values
    Requirement: the list of wieghts should be of the same length as the tuples (which represent the datapoints) 
    '''
    def __init__(self, thresholdValue:int, weightValues:list, inputValues:list) -> None:
        self.threshold = thresholdValue
        self.weights = weightValues
        self.inputs = inputValues

    def perceptron(self):
        h = 0
        for data_point in inputs:
            # print(f"The current data point for the class :\n{data_point} ")
            for actualInput, corresponding_Weight in zip(data_point, weights):
                h = h + actualInput*corresponding_Weight
                # print(f"H: {h}")
            
        if h > threshold:
            return 1
        elif h <= threshold:
            return 0
                
        
def perceptronTrainingAlgorithm(tresholdValue: int, learningRateValue:int , weightValues: list, targetValues: list, inputValues:list):
    '''
    inputValues: the list of inputs
    weightValues: the list of weights
    '''
    summation = (inputValues[0]*weightValues[0]) + (inputValues[1]*weightValues[1])
    if summation>tresholdValue:
        output = 1
    elif summation < tresholdValue:
        output = 0

    return output
    
    updatedThreshold = tresholdValue - (learningRateValue*(output-trueValue))


if __name__ == "__main__":

    testing = SimplePerceptron(thresholdValue= threshold, 
                               weightValues= weights, 
                               inputValues= inputs)

    checking = testing.perceptron()
    print(f"\OUTPUT of Perceptron is: {checking}")

# epoch = 0
# while epoch < 10:
#     for dataPoint in inputs:
#         print(f"epoch: {epoch}")
#         print(f"Data point: {dataPoint}")
#         epoch+=1
     


