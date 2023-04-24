import torch

print(torch.cuda.is_available())

DEVICE ="cuda"
n = 100
X = torch.rand(size=(100,100),device=DEVICE, requires_grad=True)
v = torch.rand(size=(100,1), device =DEVICE, requires_grad=True)
y = v.T @ (X @ v)
y.backward()
dx = v.grad


print(dx)