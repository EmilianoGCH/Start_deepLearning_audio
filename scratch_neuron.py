# import math module
import math

# define our activate funtion in order to get a result
def activate(inputs, weights):
    h = sum(map(lambda a,b: a*b, inputs,weights))
    return  1 / (1 + math.exp(-h))

if __name__ == "__main__":
# Inddispensable components of NN are inputs, weights and activate function
    inputs = [0.5, 0.3, 0.2]
    weights = [0.4,0.7,0.2]
    output = activate(inputs,weights)
    print(output)