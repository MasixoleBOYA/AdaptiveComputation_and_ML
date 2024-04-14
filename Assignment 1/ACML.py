# testing data
D = {((3, 1), 1),
     ((2, 2.5), 0), 
     ((2, 1.5), 1), 
     ((4, 3), 1), 
     ((3, 3), 0)}

# parameters randomly assigned
weights = [2, 3.5]
learningRate = 0.01
threshold = -1

# inputs and targets
inputs = [i[0] for i in D]
targets = [j[1] for j in D]

print(f"\nINPUTS: \n{inputs}")
print(f"\nTARGETS: \n{targets}")


class SimplePerceptron:
    '''
    Expects threshold value, a list weight values, list of input values
    Requirement: the list of weights should be of the same length as the tuples (which represent the datapoints) 
    '''
    def __init__(self, thresholdValue, weightValues, inputValues):
        self.threshold = thresholdValue
        self.weights = weightValues
        self.inputs = inputValues

    def perceptron(self, data_point):
        h = 0
        for actualInput, corresponding_Weight in zip(data_point, self.weights):
            h = h + actualInput * corresponding_Weight

        if h > self.threshold:
            return 1
        else:
            return 0


def perceptronTrainingAlgorithm(thresholdValue, learningRateValue, weightValues, targetValues, inputValues):
    '''
    inputValues: the list of inputs
    weightValues: the list of weights
    '''
    perc_obj = SimplePerceptron(thresholdValue=thresholdValue,
                                weightValues=weightValues,
                                inputValues=inputValues)

    for data_point, trueValue in zip(inputValues, targetValues):
        y = perc_obj.perceptron(data_point)
        thresholdValue = thresholdValue - (learningRateValue * (trueValue - y))
        for i in range(len(weightValues)):
            weightValues[i] = weightValues[i] + (learningRateValue * (trueValue - y) * data_point[i])
    return thresholdValue, weightValues


if __name__ == "__main__":
    epoch = 0
    while epoch < 10:
        threshold, weights = perceptronTrainingAlgorithm(threshold, learningRate, weights, targets, inputs)
        epoch += 1

    print("Trained Threshold:", threshold)
    print("Trained Weights:", weights)
