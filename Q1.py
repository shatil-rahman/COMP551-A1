'''
COMP 551 ASSIGNMENT 1 QUESTION 1
Name: Shatil Rahman
ID: 260606042
'''

import numpy as np
import matplotlib.pyplot as plt



#importing the data
x = np.genfromtxt('Datasets/Dataset_1_train.csv', dtype=float, delimiter=",", usecols=(0))
Y = np.genfromtxt('Datasets/Dataset_1_train.csv', dtype=float, delimiter=",", usecols=(1))

#order of the fit
order = 20

#recoding the features
X = np.c_[np.ones(50), x]

for i in range(2,order+1):
    X = np.c_[X, np.power(x,i)]





#Least squares solution
Xt = X.T
Xt_X_inv = np.linalg.inv(np.dot(Xt,X))

Xt_Y = np.dot(Xt, Y)
W = np.dot(Xt_X_inv, Xt_Y)
print W
print np.shape(X)

#Training set predictions
x_hat = np.linspace(-1.0,1.0,200)
Y_hat = np.polynomial.polynomial.polyval(x_hat, W)
#Y_hat = np.dot(X,W)



plt.plot(x,Y,'rx')
plt.plot(x_hat,Y_hat, 'b-')
plt.axis([-1.5, 1.5, -20.0, 35.0])
plt.legend(['data', 'model'])
plt.show

