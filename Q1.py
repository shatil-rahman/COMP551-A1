'''
COMP 551 ASSIGNMENT 1 QUESTION 1
Name: Shatil Rahman
ID: 260606042
'''

import numpy as np
import matplotlib.pyplot as plt



#importing the data
x = np.genfromtxt('Datasets/Dataset_1_train.csv', delimiter=",", usecols=(0))
Y = np.genfromtxt('Datasets/Dataset_1_train.csv', delimiter=",", usecols=(1))

x_val = np.genfromtxt('Datasets/Dataset_1_valid.csv', delimiter=",", usecols=(0))
Y_val = np.genfromtxt('Datasets/Dataset_1_valid.csv', delimiter=",", usecols=(1))

x_test = np.genfromtxt('Datasets/Dataset_1_test.csv', delimiter=",", usecols=(0))
Y_test = np.genfromtxt('Datasets/Dataset_1_test.csv', delimiter=",", usecols=(1))

################### Model Selection and Regularization #######################

#order and ridge regression lambda
order = 20
Lambda = 0.017

#recoding the features
X = np.c_[np.ones(50), x]

for i in range(2,order+1):
    X = np.c_[X, np.power(x,i)]


########################## Training the model ##############################
#Least squares training
Xt_X = np.dot(X.T,X)
Xt_X_inv = np.linalg.inv(np.add(Xt_X, np.eye(order + 1) * Lambda))

Xt_Y = np.dot(X.T, Y)
W = np.dot(Xt_X_inv, Xt_Y)
#print W
#print np.shape(X)


##################### MSE for Training, Validation and Test Sets #############

#Training set MSE
y_training = np.polynomial.polynomial.polyval(x,W)
error = Y - y_training 
MSE = ((error**2).sum())/(error.size)

print "Mean Squared Error on Training Set: " + str(MSE)

#Validation set MSE
Y_val_predicted = np.polynomial.polynomial.polyval(x_val, W)
error_val = Y_val - Y_val_predicted
MSE_val = ((error_val**2).sum())/(error_val.size)

print "Mean Squared Error on Validation Set: " + str(MSE_val)

#Test set MSE
Y_test_predicted = np.polynomial.polynomial.polyval(x_test, W)
error_test = Y_test - Y_test_predicted
MSE_test = ((error_test**2).sum())/(error_test.size)

print "Mean Squared Error on Test Set: " + str(MSE_test)

###################### Plotting and Visualization ############################
#Visualizing the fit
x_fit = np.linspace(-1.0,1.0,200)
Y_fit = np.polynomial.polynomial.polyval(x_fit, W)

#Plotting training results
plt.figure(figsize=[10,10])
plt.subplot(311)
plt.plot(x,Y,'rx')
plt.plot(x_fit,Y_fit, 'b-')
plt.axis([-1.5, 1.5, -20.0, 35.0])
plt.title('Linear Regression by SGD on Training Data')
plt.legend(['Training data', 'model'], loc=2)


#Plotting validation results
plt.subplot(312)
plt.plot(x_val,Y_val,'rx')
plt.plot(x_fit,Y_fit, 'b-')
plt.axis([-1.5, 1.5, -20.0, 35.0])
plt.title('Linear Regression by SGD on Validation Data')
plt.legend(['Validation data', 'model'], loc=2)

#Plotting test results
plt.subplot(313)
plt.plot(x_test,Y_test,'rx')
plt.plot(x_fit,Y_fit, 'b-')
plt.axis([-1.5, 1.5, -20.0, 35.0])
plt.title('Linear Regression by SGD on Test Data')
plt.legend(['Test data', 'model'], loc=2)
plt.show

