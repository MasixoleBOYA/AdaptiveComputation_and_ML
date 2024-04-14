import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_hidden = weights_input_hidden
        self.bias_hidden = bias_hidden
        self.weights_hidden_output = weights_hidden_output
        self.bias_output = bias_output

    def set_weights_and_biases(self, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output
        self.bias_hidden = bias_hidden
        self.bias_output = bias_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input_list):
        #Input layer fedforwad into the hidden layer
        hidden_inputs = np.dot(input_list, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        #Hidden layer now is being fed into output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)

        # print(f"\nFinal outputs: {final_outputs}")

        return final_outputs
    #the loss
    def sum_of_squares_loss(self, output_values, target_values):
        return 0.5 * np.sum((output_values - target_values) ** 2)
    
    def backpropagation(self, input_list, target_values, learning_rate):
        #Convert weights into NumPy array
        weights_hidden_output_np = np.array(self.weights_hidden_output)

        #perfoming the feed forward step here also
        hidden_inputs = np.dot(input_list, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        #its output
        final_inputs = np.dot(hidden_outputs, weights_hidden_output_np) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)

        #Deltas for output layer
        output_errors = target_values - final_outputs
        output_deltas = output_errors * final_outputs * (1 - final_outputs)

        #Deltas for input layer hidden layer
        hidden_errors = np.dot(output_deltas, weights_hidden_output_np.T)
        hidden_deltas = hidden_errors * hidden_outputs * (1 - hidden_outputs)

        #Updating the weight values and the bias values
        self.weights_hidden_output += learning_rate * np.outer(hidden_outputs, output_deltas)
        self.bias_output += learning_rate * output_deltas

        self.weights_input_hidden += learning_rate * np.outer(input_list, hidden_deltas)
        self.bias_hidden += learning_rate * hidden_deltas

#The actual code

input_nodes = 4
hidden_nodes = 8
output_nodes = 3

#weights matrix
weights_from_InputToHidden = [[1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1]]

bias_hidden = [1, 1, 1, 1, 1, 1, 1, 1]

weights_from_HiddenToOutput = [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]

bias_output = [1, 1, 1]
    
#an instance of the network
neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, weights_from_InputToHidden, weights_from_HiddenToOutput, bias_hidden, bias_output)

testing_input_list =[]

for i in range(1,8):
    inputting = float(input())
    testing_input_list.append(inputting)

#Standard input
input_values = [testing_input_list[i] for i in range(4)]
target_values = [testing_input_list[i] for i in range(4, 7)]

#Feedforward the input values
output_values_before = neural_network.feedforward(input_values)
loss_before = neural_network.sum_of_squares_loss(output_values_before, target_values)
print(round(loss_before, 4))

#One iteration of backpropagation
neural_network.backpropagation(input_values, target_values, learning_rate=0.1)

#Feedforward the input values again
output_values_after = neural_network.feedforward(input_values)
loss_after = neural_network.sum_of_squares_loss(output_values_after, target_values)
print(round(loss_after, 4))
   