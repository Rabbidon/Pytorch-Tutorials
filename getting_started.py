import torch

# Initialise matrix with arbitrary starting values
print(torch.empty(5,3))

# Initialise matrix with random values in [0,1]
print(torch.rand(5,3))

# Initialise matrix with zeros, and enforce type long
print(torch.zeros(5, 3, dtype=torch.long))

# Construct tensor directly from data
print([5.5,3])
