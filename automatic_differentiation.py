import torch

print("Since tensors are so commonly used to deal with backpropogation, this functionality is built into torch tensors\n")

print("When we create a tensor we can turn on gradient tracking. Any objects we create from this tensor will then also track gradient.")

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

print("Torch tracks the mathematical relations between objects and does backpropogation for you, so you can get gradients. When you call backward() on a tensor, it automatically calculates all gradients w.r.t all objects it is descended from.\n")

out.backward()

print("We can then calculate derivatives w.r.t a specific object (say x) by calling x.grad")

print(x.grad)

print("We can only call backward on a scalar, so if we have a tensor-valued function we can't use it directly. We have to first take the inner product with a tensor the same shape as the output tensor")

x = torch.tensor([[10,1,1,],[10,1,1]], dtype=torch.float, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([[0.1, 1.0, 0.0001],[0.1,1,0.0001]], dtype=torch.float)

y.backward(v)
print(x.grad)
