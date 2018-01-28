'''
COMP 551 ASSIGNMENT 1 QUESTION 3
Name: Shatil Rahman
ID: 260606042
'''
import numpy as np
import matplotlib.pyplot as plt

def train(X,Y,Lambda):
    '''
    -Linear Regression with L2 regularization, with Input matrix X, 
     Output vector Y, and regularization constant Lambda
    -Returns array of learned weights(parameters)
    '''
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
    '''
    -Loads the CandC-data in the folder shown
    -IMPORTANT: If for whatever reason the data isnt found, fix path here
    -Loads all the training data, from set 1 - 5 into examples_train
    -examples_train is a list of tuples, each tuple such that tuple[0] is the
     X-data and tuple[1] is the output Y data
    -Same thing for the test data, puts it into examples_test
    -Column of ones is added at the beginning of the X-data!
    '''
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
    
def iterateLambda(start, step, plot):
    '''
    -Function designed to try a range of lambdas, defined by start, and 
     increased by step, and will return the lambda which resulted in lowest
     average MSE
    -For each Lambda, trains 5 models for the 5 datasets, calculates the 
     MSE for each model, takes the average, keeping track of the minimum Lambda
     and minimum MSE
    -For a plot of avg MSE vs Lambda, call function with plot=1
    '''
    examples_train = []
    examples_test = []
    Avg_MSE = []
    Lambda = start
    Lambda_list = []
    
    min_MSE = 10000.0
    min_Lambda = start
    
    loadData(examples_train, examples_test)
    
    #Iterating over the lambdas
    for j in range(0,20):
        Models = []
        MSE_list = []
        
        #Training 5 models for the 5 datasets
        for i in range(0,5):
            example = examples_train[i]
            test_example = examples_test[i]
            
            Models.append(train(example[0], example[1], Lambda))
            MSE_test = MSE(Models[i], test_example[0], test_example[1])
            MSE_list.append(MSE_test)
            #print "MSE for Set 1: " + str(MSE_test)
            
        MSE_list = np.array(MSE_list)
        Avg = np.average(MSE_list)
        
        #keeping track of the minimums
        if Avg < min_MSE:
            min_MSE = Avg
            min_Lambda = Lambda
            
            
        Avg_MSE.append(Avg)
        Lambda_list.append(Lambda)
        Lambda = Lambda + step
    
    #Plotting if desired
    if plot:
        plt.figure(figsize=[10,10])
        plt.plot(Lambda_list, Avg_MSE, 'b-')
        plt.show
        
    return [min_Lambda, min_MSE]

        
    
    
    
    
########################  Main Code  #################################

examples_train = []
examples_test = []

loadData(examples_train, examples_test)

Lambda = 1.27
Models = [] #Each item in Models is a set of weights learned
MSE_list = [] #i-th entry of MSE_list contains the MSE for model for dataset i

#Learning the model for each dataset
for i in range(0,5):
    example = examples_train[i]
    test_example = examples_test[i]
        
    Models.append(train(example[0], example[1], Lambda))
    MSE_test = MSE(Models[i], test_example[0], test_example[1])
    MSE_list.append(MSE_test)
    print "MSE for Set " +  str(i+1) + ":   " + str(MSE_test)
        
MSE_list = np.array(MSE_list)
Avg_MSE = np.average(MSE_list)

print "Average MSE:     " + str(Avg_MSE)



##################### Feature Reduction ################################
'''
- Using the results of the L2- Regularization, features with weights less
  than a certain limit are removed, and the model is relearned and tested    
'''
reduced_Models = []
reduced_MSE_list = []
useful_weights = []
for j in range(0,5):
    useful_weights.append( [i for i, W in enumerate(Models[j]) if abs(W) >=0.1])
    example = examples_train[j]
    example_test = examples_test[j]
    #X_test = example[0]
    X_reduced = example[0][:,useful_weights[j]]
    W_reduced = train(X_reduced, example[1], 0.0)
    
    reduced_Models.append(W_reduced)
    
    X_test_reduced = example_test[0][:,useful_weights[j]]
    MSE_reduced = MSE(W_reduced,X_test_reduced, example_test[1])
    reduced_MSE_list.append(MSE_reduced)
    print "MSE on Test Set " + str(j+1) +" with " + str(W_reduced.shape[0]) + " features: " + str(MSE_reduced)

reduced_MSE_list = np.array(reduced_MSE_list)
Avg_reduced_MSE = np.average(reduced_MSE_list)
print "Average reduced MSE:     " + str(Avg_reduced_MSE)

best_useful_features = np.array(useful_weights[3]).reshape(reduced_Models[3].shape[0],1)
feature_no_and_weight = np.concatenate((best_useful_features,reduced_Models[3]), axis=1)

sorted_features_by_weight = feature_no_and_weight[np.argsort(np.absolute(feature_no_and_weight[:,1]))]





    







        
    
