# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#concatenating rows
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(x_1)
print(y_1)
print(z_1)

# Concatenate columns:
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
# second arg specifies which axis to concat along
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)


print("\nVARIABLE DEMO\n")
# Variables wrap tensor objects
x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
# You can access the data with the .data attribute
print(x.data)

# You can also do all the same operations you did with tensors with Variables.
y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
z = x + y
print(z.data)

# BUT z knows something extra.
print(z.grad_fn)

# Lets sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)


print ("\n AFFINE  MAPS \n")


torch.manual_seed(1)
lin = nn.Linear(5, 3)  # maps from R^5 to R^3, parameters A, b
# data is 2x5.  A maps from 5 to 3... can we map "data" under A?
data = autograd.Variable(torch.randn(2, 5))
print(lin(data))  # yes

m = nn.Linear(20, 30)
input = autograd.Variable(torch.randn(128, 20))
output = m(input)
print(output.size())