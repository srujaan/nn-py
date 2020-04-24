inputs = [1.43, 5.23, 2.35, 7.34]

weights = [[4, 2, 1.4, -2.5],
            [0.3, 2.1, 4, 9],
            [-2.4, 4, 2, -3.1]]

bias = [0.4, 3, 0.2]

layer_output = []  # output of current layer

for neuron_weight, neuron_bias in zip(weights, bias):
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)


