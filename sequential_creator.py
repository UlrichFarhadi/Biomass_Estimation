# File created by Ulrich Farhadi, used in other university projects. (Credit given to avoid plaigarism issues)

import torch
import numpy as np

def make_model(input, hidden, output, activation=torch.nn.ReLU()):
    # example input = 12, 3 hidden layers of 30, output = 3 --> 12, [30, 30, 30], 3 --> 12-30, 30-30, 30,30, 30-3 --> Loop should be (hidden + 1) iterations
    
    list_seq = [torch.nn.Linear(input, hidden[0]), activation] # Initial layer

    for i in range(len(hidden) - 1): # Hidden Layers
        list_seq.append(torch.nn.Linear(hidden[i], hidden[i+1]))
        list_seq.append(activation)

    list_seq.append(torch.nn.Linear(hidden[len(hidden) - 1], output))
    list_seq.append(activation)
    
    #print(list_seq)
    #print(*list_seq)
    model = torch.nn.Sequential(*list_seq) # Need to unpack the list to use it in sequential
    return model

# Other way of defining the model manually
# model = torch.nn.Sequential(torch.nn.Linear(12, 20),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(20, 30),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(30, 20),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(20, 10),
#                             torch.nn.ReLU(),
#                             torch.nn.Linear(10, 3),
#                             torch.nn.ReLU()
#                             )