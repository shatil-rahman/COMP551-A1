'''
COMP 551 ASSIGNMENT 1 QUESTION 3
Name: Shatil Rahman
ID: 260606042
'''
import numpy as np
import matplotlib.pyplot as plt

def train(X,Y,Lambda):
    #Least squares training
    Xt_X = np.dot(X.T,X)
    Xt_X_inv = np.linalg.inv(np.add(Xt_X, np.eye(Xt_X.shape[0]) * Lambda))
    
    Xt_Y = np.dot(X.T, Y)
    W = np.dot(Xt_X_inv, Xt_Y)
    return W

def MSE(W,X,Y):
    #Calculate mean square error
    y_hat = np.dot(X,W)
    error = Y - y_hat 
    MSE = ((error**2).sum())/(error.size)
    return MSE
 
def loadData(examples_train, examples_test):
    for i in range(1,6):
        s_train = "Datasets/CandC-train" + str(i) + ".csv"
        s_test = "Datasets/CandC-test" + str(i) + ".csv"
        
        X = np.loadtxt(s_train, dtype=float, delimiter=',', ndmin=2)
        o = np.ones(X.shape[0]).reshape(X.shape[0],1)
        X = np.concatenate((o,X), axis=1)
        examples_train.append([X[:,:-1], X[:,-1].reshape(X.shape[0],1)])
        
        X2 = np.loadtxt(s_test, dtype=float, delimiter=',', ndmin=2)
        o = np.ones(X2.shape[0]).reshape(X2.shape[0],1)
        X2 = np.concatenate((o,X2), axis=1)
        examples_test.append([X2[:,:-1], X2[:,-1].reshape(X2.shape[0],1)])
    
        


#############################Load Data #######################################
'''
examples_train = []
examples_test = []

for i in range(1,6):
    s_train = "Datasets/CandC-train" + str(i) + ".csv"
    s_test = "Datasets/CandC-test" + str(i) + ".csv"
    
    X = np.loadtxt(s_train, dtype=float, delimiter=',', ndmin=2)
    o = np.ones(X.shape[0]).reshape(X.shape[0],1)
    X = np.concatenate((o,X), axis=1)
    examples_train.append([X[:,:-1], X[:,-1].reshape(X.shape[0],1)])
    
    X2 = np.loadtxt(s_test, dtype=float, delimiter=',', ndmin=2)
    o = np.ones(X2.shape[0]).reshape(X2.shape[0],1)
    X2 = np.concatenate((o,X2), axis=1)
    examples_test.append([X2[:,:-1], X2[:,-1].reshape(X2.shape[0],1)])
'''  
    
##########################Training the models #################################

examples_train = []
examples_test = []

loadData(examples_train, examples_test)

Avg_MSE = []
Lambda = 1.2
Lambda_list = []

for j in range(0,20):
    Models = []
    MSE_list = []
    for i in range(0,5):
        example = examples_train[i]
        test_example = examples_test[i]
        
        Models.append(train(example[0], example[1], Lambda))
        MSE_test = MSE(Models[i], test_example[0], test_example[1])
        MSE_list.append(MSE_test)
        #print "MSE for Set 1: " + str(MSE_test)
        
    MSE_list = np.array(MSE_list)
    Avg_MSE.append(np.average(MSE_list))
    Lambda_list.append(Lambda)
    Lambda = Lambda + 0.01
    
plt.figure(figsize=[10,10])
plt.plot(Lambda_list, Avg_MSE, 'b-')
plt.show


    







        
    
