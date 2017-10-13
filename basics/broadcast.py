import  numpy as  np

A = np.array([[1.0, 2.0, 3.0, 4.0],
              [1.0, 2.0, 3.0, 5.0],
              [1.0, 2.0, 3.0, 6.0],
              ])

print(A.ndim)

print(A)

cal = A.sum(axis=0)

print(cal)

percentage = 100 * A / cal.reshape(1,4)

print(type(cal.reshape(1,4)))

print(A.shape)
print(cal.reshape(1, 4).shape)

print(percentage.shape)

T = np.mat(A)
print(T)
print(T.T)
print(type(T))
print(T.I)


a = 5

print(a**2)