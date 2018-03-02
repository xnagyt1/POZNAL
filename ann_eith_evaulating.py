
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

def biggest(a, y, z):
    Max = a
    if y > Max:
        Max = y    
    if z > Max:
        Max = z
        if y > z:
            Max = y
    return Max

# Importing the dataset
datasetX = pd.read_csv('ADAS_ADNIGO23.csv')
datasetY = pd.read_csv('TADPOLE_D1_D2.csv')
full_df = pd.merge(datasetX, datasetY,  how='left', left_on=['RID','VISCODE2','Phase'], right_on = ['RID','VISCODE','COLPROT'])
full_df = full_df[full_df['DX_bl'].notnull()]
# VISCODE,Q1SCOR,Q4SCOR,Q5SCOR,Q6SCOR,Q7SCOR,Q8SCOR,Q9SCOR,Q10SCOR,Q11SCOR,Q12SCOR,TOTSCOR,Q13SCOR,TOTAL13,DX_bl,AGE,PTGENDER,PTEDUCAT,PTMARRY
X = full_df.iloc[:,[5,16,20,26,30,33,50,53,103,105,107,109,111,113,118,119,129,131,132,133,136]]
Y = X.iloc[:,16]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:,4])
onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

X_Y_train = np.concatenate((X_train,np.reshape(Y_train,(len(Y_train),1))),axis=1)

x_0 = np.where(X_Y_train[:,5] == 0)
x_0 = X_Y_train[x_0]

x_1 = np.where(X_Y_train[:,5] == 1)
x_1 = X_Y_train[x_1]

x_2 = np.where(X_Y_train[:,5] == 2)
x_2 = X_Y_train[x_2]

max = biggest(len(x_0),len(x_1),len(x_2))

for i in range(0,max):
    if i>=len(x_0):
        x_0 = np.vstack((x_0,x_0[random.randint(0,len(x_0)-1)]))
    if i>=len(x_1):
        x_1 = np.vstack((x_1,x_1[random.randint(0,len(x_1)-1)]))
    if i>=len(x_2):
        x_2 = np.vstack((x_2,x_2[random.randint(0,len(x_2)-1)]))
        
x_0 = np.vstack((x_0,x_1))
x_0 = np.vstack((x_0,x_2))
X_Y_train = x_0

X_train = X_Y_train[:,:-1]
Y_train = X_Y_train[:,-1]
        

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_classifier(units,optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu',input_dim = 5))
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units = 3,kernel_initializer='uniform',activation='softmax'))
    classifier.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25,32], 'epochs' : [500], 'units': [5,10,15], 'optimizer' : ['rmsprop','adam']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

classifier = build_classifier(15,'adam')

# Fitting the ANN to the Training set
classifier.fit(X_train,Y_train,batch_size = 25, epochs = 500)

# Predicting the Test set results
Y_pred = classifier.predict_classes(X_test)
Y_pred = (Y_pred>0.5)

# Making the Confucion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
