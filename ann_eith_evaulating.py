# Importing the libraries
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

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
"""

corelogramTests = X_full.iloc[:,[1,2,3,4,15,16]]
corelogramAgeGenderEducationMaried = X_full.iloc[:,[15,17,18,19,16]]
sns.pairplot(corelogramAgeGenderEducationMaried)
plt.show()

le = LabelEncoder()
le.fit(X_full['DX_bl'])
X_full['DX_bl'] = le.transform(X_full['DX_bl'])


X_full.iloc[:,16].apply(LabelEncoder().fit_transform)

sns.pairplot(corelogramAgeGenderEducationMaried, kind="scatter", hue="DX_bl", markers=["o", "o", "o","o","o"], palette="Set2")
plt.show()
 
# right: you can give other arguments with plot_kws.
sns.pairplot(df, kind="scatter", hue="species", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()
"""

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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping

def build_classifier(units1,units2,optimizer,dropout):
    classifier = Sequential()
    classifier.add(Dense(units = units1,kernel_initializer='uniform',activation='relu',input_dim = 33))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units2,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units2,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = units1,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = dropout))
    classifier.add(Dense(units = 4,kernel_initializer='uniform',activation='softmax'))
    classifier.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [32], 'epochs' : [500], 'units1': [128,256,512], 'units2': [128,256,512], 'optimizer' : ['adam'],'dropout':[0.5],'callbacks':[[EarlyStopping(monitor='acc', patience=5)]]}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


#classifier = build_classifier(200,300,'adam',0.5)
# Fitting the ANN to the Training set
#classifier.fit(X_train,Y_train,batch_size = 32, epochs = 500,callbacks=[EarlyStopping(monitor='acc', patience=10)])


# Predicting the Test set results
Y_pred = grid_search.predict_classes(X_test)
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

# rozhodovacie stromi a ich rozne verzie Stochastic gradien boosted tree


        
        
        