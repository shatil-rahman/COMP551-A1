'''
COMP 551 ASSIGNMENT 1 QUESTION 2
Name: Shatil Rahman
ID: 260606042
'''

import numpy as np
import matplotlib.pyplot as plt

#importing the data
x = np.genfromtxt('Datasets/Dataset_2_train.csv', dtype=float, delimiter=",", usecols=(0))
x = np.array([x]).T
Y = np.genfromtxt('Datasets/Dataset_2_train.csv', dtype=float, delimiter=",", usecols=(1))
Y = np.array([Y]).T

x_val = np.genfromtxt('Datasets/Dataset_2_valid.csv', dtype=float, delimiter=",", usecols=(0))
x_val = np.array([x_val]).T
Y_val = np.genfromtxt('Datasets/Dataset_2_valid.csv', dtype=float, delimiter=",", usecols=(1))
Y_val = np.array([Y_val]).T

x_test = np.genfromtxt('Datasets/Dataset_2_test.csv', dtype=float, delimiter=",", usecols=(0))
x_test = np.array([x_test]).T
Y_test = np.genfromtxt('Datasets/Dataset_2_test.csv', dtype=float, delimiter=",", usecols=(1))
Y_test = np.array([Y_test]).T

############Training the model, usign Stochastic Gradient Descent ###########
#Initial Guess, W0, step size

W_prev = np.array([[5.0], [9.0]])
W = np.array([[5.0], [9.0]])

alpha = 0.01

#SGD:
training_MSE_epoch = []
validation_MSE_epoch = []

iteration = 0
while True:
    
    iteration = iteration + 1   
    
    #One epoch of the SGD Algorithm
    for i in range(0,300):
        x2 = np.array([[1.0], x[i]]).T
        y_hat = np.dot(x2, W)

        W[0] = W[0] - alpha * ( y_hat - Y[i])
        W[1] = W[1] - alpha * ( y_hat - Y[i])* x[i]

    #Calculating the MSE in the epoch
    y_epoch_train = np.polynomial.polynomial.polyval(x,W, tensor=False)
    y_epoch_val = np.polynomial.polynomial.polyval(x_val,W, tensor=False)
    
    error_epoch_val = Y_val - y_epoch_val
    error_epoch_train = Y - y_epoch_train
    
    cur_MSE_val = ((error_epoch_val**2).sum())/(error_epoch_val.size)
    cur_MSE_train = ((error_epoch_train**2).sum())/(error_epoch_train.size)    
    
    validation_MSE_epoch.append(cur_MSE_val)
    training_MSE_epoch.append(cur_MSE_train)
    
    #Stopping condition
    if np.linalg.norm(np.subtract(W, W_prev)) < 0.001:
        break
    else:
        W_prev = np.copy(W)

print "No of learning iterations: " + str(iteration)

##################### MSE for Training, Validation and Test Sets #############

#Training set MSE

y_training = np.polynomial.polynomial.polyval(x,W,tensor=False)
error = Y - y_training 
MSE = ((error**2).sum())/(error.size)

print "Mean Squared Error on Training Set: " + str(MSE)

#Validation set MSE
Y_val_predicted = np.polynomial.polynomial.polyval(x_val, W, tensor=False)
error_val = Y_val - Y_val_predicted
MSE_val = ((error_val**2).sum())/(error_val.size)

print "Mean Squared Error on Validation Set: " + str(MSE_val)

#Test set MSE
Y_test_predicted = np.polynomial.polynomial.polyval(x_test, W, tensor=False)
error_test = Y_test - Y_test_predicted
MSE_test = ((error_test**2).sum())/(error_test.size)

print "Mean Squared Error on Test Set: " + str(MSE_test)


###################### Plotting and Visualization ############################
#Visualizing the fit
x_fit = np.linspace(0,2,200)
Y_fit = np.polynomial.polynomial.polyval(x_fit, W)

#Plotting learning curve
plt.figure(figsize=[10,10])
plt.subplot(411)
plt.plot(training_MSE_epoch, 'b-')
plt.plot(validation_MSE_epoch, 'r-')
plt.ylabel('MSE')
plt.xlabel('No of epochs')
plt.title('SGD Learning Curve')
plt.legend(['Training', 'Validation'])

#Plotting training results

plt.subplot(412)
plt.plot(x,Y,'rx')
plt.plot(x_fit,Y_fit.T, 'b-')
plt.axis([0.0, 1.5, -20.0, 35.0])
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linear Regression by SGD on Training Data')
plt.legend(['Training data', 'model'], loc=2)


#Plotting validation results
plt.subplot(413)
plt.plot(x_val,Y_val,'rx')
plt.plot(x_fit,Y_fit.T, 'b-')
plt.axis([0.0, 1.5, -20.0, 35.0])
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linear Regression by SGD on Validation Data')
plt.legend(['Validation data', 'model'], loc=2)

#Plotting test results
plt.subplot(414)
plt.plot(x_test,Y_test,'rx')
plt.plot(x_fit,Y_fit.T, 'b-')
plt.axis([0.0, 1.5, -20.0, 35.0])
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linear Regression by SGD on Test Data')
plt.legend(['Test data', 'model'], loc=2)
plt.show

print "Weights(W0 ... Wn) are: " + str(W)

