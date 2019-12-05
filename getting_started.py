import torch

print("The first tutorial in \"Deep Learning with PyTorch: A 60 Minute Blitz\"")

print("\n")

print("Initialise matrix with arbitrary starting values")
x = torch.empty(5, 3)
print(x)

print("Initialise matrix with random values in [0,1]")
x = torch.rand(5, 3)
print(x)

print("Initialise matrix with zeros, and enforce type long")
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

print("Construct tensor directly from data (also convert existing arrays to tensors")

x=torch.tensor([5.5,3])
print(x)

print("Construct new tensor with properties of given tensor")

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = x.new_ones(5, 3, dtype=torch.float)
print(x)

print("Get the size of x")

print(x.size())

print("Addition Syntax")

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

print("Outputting addiiton result into an existing tensor")

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

print("In-place addition")

y.add_(x)
print(y)

print("You can usee all numpy slicing operations")

print(x[:, 1])

print("Resizing is possible using torch.view")

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

print("Getting a single-element tensor as a number")

x = torch.randn(1)
print(x)
print(x.item())

print("Convert tensorflow tensor to numpy array")

a=torch.ones(1,5)
b=a.numpy()
print(a)
print(b)

print("Editing the underlying tensor will also edit any numpy conversion attached to it")

a.add_(a)
print(a)
print(b)

print("Tensors have an assigned device that handles operations on them. We can change it using the .to method")

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
