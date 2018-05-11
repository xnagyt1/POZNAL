# Importing the libraries
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE

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
full_df = pd.merge(datasetX, datasetY,  how='left',
                   left_on=['RID','VISCODE2','Phase'],
                   right_on = ['RID','VISCODE','COLPROT'])
full_df = full_df[full_df['DX_bl'].notnull()]
full_df = full_df[full_df['DX_bl']!='SMC']


# VISCODE,Q1SCOR,Q4SCOR,Q5SCOR,Q6SCOR,Q7SCOR,Q8SCOR,Q9SCOR,Q10SCOR,Q11SCOR,Q12SCOR,TOTSCOR,Q13SCOR,TOTAL13,DX_bl,AGE,PTGENDER,PTEDUCAT,PTMARRY
X = full_df.iloc[:,[5,16,20,26,30,33,50,53,103,105,107,109,111,113,118,119,129,131,132,133,136]]

# Dropping nan values
X = X.replace('-4', np.nan)
X = X.dropna(axis=0, how='any')
X_full = X



Y = X.iloc[:,16]
X = X.loc[:, X.columns != 'DX_bl']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X = X.values
Y = Y.values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,19] = labelencoder_X.fit_transform(X[:,19])

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_X = LabelEncoder()
X[:,17] = labelencoder_X.fit_transform(X[:,17])

onehotencoder = OneHotEncoder(categorical_features=[19])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
onehotencoder = OneHotEncoder(categorical_features=[21])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
onehotencoder = OneHotEncoder(categorical_features=[5])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

# Oversampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(X_train, Y_train)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(units,optimizer,dropout):
    classifier = Sequential()
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu',input_dim = 33))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = 5,kernel_initializer='uniform',activation='softmax'))
    classifier.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier


classifierANN = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [32], 'epochs' : [500], 'units': [200], 'optimizer' : ['adam'],'dropout':[0.5],'callbacks':[[EarlyStopping(monitor='acc', patience=20)]]}
grid_search = GridSearchCV(estimator = classifierANN, param_grid = parameters, scoring = 'accuracy')
#grid_search = grid_search.fit(X_train,Y_train)
#classifierANN.fit(X_train,Y_train,inputs = parameters)

from sklearn.ensemble import GradientBoostingClassifier
classifierGBT = GradientBoostingClassifier(random_state=0, learning_rate = 0.2, max_depth=8,min_samples_leaf=5,n_estimators=200)
#rfe = RFE(classifierGBT,25)
#classifierGBT = classifierRF.fit(X_train,Y_train)

from sklearn.ensemble import VotingClassifier
classifier = VotingClassifier(estimators=[('ann', grid_search), ('gbt', classifierGBT)], 
                        voting='hard', weights=[1,2])
grid_search.fit(X_train, Y_train)


Y_pred = grid_search.predict(X_test)
#Y_pred = (Y_pred>0.5)

# Making the Confucion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

# Makeing classification report
from sklearn.metrics import classification_report
target_names = ['CN', 'EMCI', 'LMCI','AD']
cr = classification_report(Y_test, Y_pred, target_names=target_names)

# Counting BMI, accuracy
countAll = 0
countTrue = 0
BMI = 0
BMIRow = 0
BMIValue = 0

for i in range(0,len(cm)):
    countRow = cm[i,:]
    BMIRow = 0
    for j in range(0,len(cm)):
        BMIRow += cm[i,j]
        if(i==j):
            BMIValue = cm[i,j]
            countTrue += cm[i,j]
        countAll += cm[i,j]
    BMI+=BMIValue/BMIRow


BMI = BMI/len(cm)
accuracy = countTrue/countAll

#classifierANN.feature_importances


        
        
        