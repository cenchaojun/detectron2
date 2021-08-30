import numpy as np

r = np.random.rand(2)
print(r)
x_out = lambda x: x * r[0] + r[1] * 10
print(x_out(r))

a=lambda x:x*x
print(a(3))

def lab(x):
    x= x * r[0] + r[1] * 10
    return x

print(lab(r))