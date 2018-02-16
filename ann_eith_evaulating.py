
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
datasetX = pd.read_csv('ADAS_ADNI1.csv')
datasetY = pd.read_csv('TADPOLE_D1_D2.csv')
full_df = pd.merge(datasetX, datasetY,  how='left', left_on=['RID','VISCODE'], right_on = ['RID','VISCODE'])
full_df = full_df[full_df['DX_bl'].notnull()]

#X = full_df.iloc[:,[1,3,10,11,12,92,93]].values
X = full_df.iloc[:,[10,11,12,92,93]].values
Y = full_df.iloc[:,[90]].values

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


X_Y_train = np.concatenate((X_train,Y_train),axis=1)


X_0 = []
X_1 = []
X_2 = []

for a in range(0,len(Y_train)):
    if Y_train[a] == 0:
        X_0 = X_0.append(X_train[a])
    elif Y_train[a] == 1:
        X_1 = X_1.append(X_train[a])
    else:
        X_2 = X_2.append(X_train[a])
        

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
parameters = {'batch_size' : [25,32], 'epochs' : [500], 'units': [5,6,7,8], 'optimizer' : ['rmsprop','adam']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

classifier = build_classifier(50,'rmsprop')

# Fitting the ANN to the Training set
classifier.fit(X_train,Y_train,batch_size = 32, epochs = 500)

# Predicting the Test set results
Y_pred = classifier.predict_classes(X_test)
Y_pred = (Y_pred>0.5)

# Making the Confucion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
