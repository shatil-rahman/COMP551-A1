'''
COMP 551 ASSIGNMENT 1 QUESTION 1
Name: Shatil Rahman
ID: 260606042
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

feature_1 = []
output_1 = []

with open('Datasets/Dataset_1_test.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x = float(row[0])
        y = float(row[1])
        feature_1.append(x)
        output_1.append(y)
        
X = np.array(feature_1)
Y = np.array(output_1)

plt.plot(X,Y,'bx')
plt.show
    
