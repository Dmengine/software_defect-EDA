# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 02:20:18 2021

@author: me
"""


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from WOA import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt


# load data
data  = pd.read_csv('CM1.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in 
N    = 10    # number of particles
T    = 10   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

# mdl       = KNeighborsClassifier(n_neighbors = k) 
# mdl.fit(x_train, y_train)

# # # accuracy
# y_pred    = mdl.predict(x_valid)

from sklearn.tree import DecisionTreeClassifier
mdl1=  DecisionTreeClassifier()
mdl1.fit(x_train, y_train)

# # accuracy
y_pred1    = mdl1.predict(x_valid)
# Acc       =( np.sum(y_valid == y_pred)  / num_valid) *100
Acc1      =( np.sum(y_valid == y_pred1)  / num_valid) *100
print("Accuracy:", Acc1)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
Accuracies=[Acc1]

Classifiers=["WOAID3"]
# plt.plot(Classifiers,Accuracies,'m')  
# plt.scatter(Classifiers,Accuracies )
plt.style.use('fivethirtyeight')
plt.bar(Classifiers,Accuracies ,color=['purple'])  
plt.title("Software Faults Prediction for KC2")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")
plt.show()

