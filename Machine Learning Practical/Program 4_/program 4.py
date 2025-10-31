import numpy as np

# Input and output data
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# Normalization
X = X / np.amax(X, axis=0)
y = y / 100

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

# Neural network parameters
epoch = 5000
lr = 0.1
inputlayer_neurons = 2
hiddenlayer_neurons = 3
output_neurons = 1

# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Training loop
for i in range(epoch):
    # Forward propagation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    
    hlayer_act = sigmoid(hinp)
    
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout                        
    
    
    
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad

    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Updating weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

# Output
print("Input: \n", X)
print("Actual Output: \n", y)
print("Predicted Output: \n", output)