import torch

# Initialise matrix with arbitrary starting values
print(torch.empty(5,3))

# Initialise matrix with random values in [0,1]
print(torch.rand(5,3))

# Initialise matrix with zeros, and enforce type long
print(torch.zeros(5, 3, dtype=torch.long))

# Construct tensor directly from data (also convert existing arrays to tensors)

x=torch.tensor([5.5,3])
print(x)

# COnstruct new tensor with properties of given tensorN

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = x.new_ones(5, 3, dtype=torch.float)
print(x) 

# Convert tensorflow tensor to numpy array

print(x.numpy())
