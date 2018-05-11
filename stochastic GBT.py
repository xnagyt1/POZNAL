# Importing the libraries
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import RFE

# Importing the dataset
datasetX = pd.read_csv('ADAS_ADNIGO23.csv')
datasetY = pd.read_csv('TADPOLE_D1_D2.csv')
full_df = pd.merge(datasetX, datasetY,  how='left', left_on=['RID','VISCODE2','Phase'], right_on = ['RID','VISCODE','COLPROT'])
full_df = full_df[full_df['DX_bl'].notnull()]
#full_df = full_df[full_df['DX_bl']!='SMC']

# VISCODE,Q1SCOR,Q4SCOR,Q5SCOR,Q6SCOR,Q7SCOR,Q8SCOR,Q9SCOR,Q10SCOR,Q11SCOR,Q12SCOR,TOTSCOR,Q13SCOR,TOTAL13,DX_bl,AGE,PTGENDER,PTEDUCAT,PTMARRY
X = full_df.iloc[:,[5,16,20,26,30,33,50,53,103,105,107,109,111,113,118,119,129,131,132,133,136]]


# Dropping nan values
X = X.replace('-4', np.nan)
X = X.dropna(axis=0, how='any')
X_full = X

Y = X.iloc[:,16]
X = X.loc[:, X.columns != 'DX_bl']



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

#selected features
#X = X[:,[1,3,4,5,6,7,8,9,10,11,12,13,15,16,19,20,22,23,25,26,28,29,30,31,32]]

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

#BUild classifier

classifier = GradientBoostingClassifier(random_state=0, learning_rate = 0.2, max_depth=8,min_samples_leaf=5,n_estimators=200)
#classifier = GradientBoostingClassifier(random_state=0)
classifier = classifier.fit(X_train,Y_train)
'''
parameters = {'learning_rate': [0.2, 0.1, 0.05, 0.02, 0.01], rychlost ucenia
              'max_depth': [4, 6, 8,10],
              'min_samples_leaf': [10,20, 50,100,150],
              'n_estimators': [100,200,400, 800] pocet stromov
              #'max_features': [1.0, 0.3, 0.1] 
              }
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = 4)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
'''
#best parameters 0.1, 8, 20, 400
'''
rfe = RFE(classifier,25)
rfe = rfe.fit(X_train, Y_train)
print(rfe.support_)
print(rfe.ranking_)
    '''      
                   
Y_pred = classifier.predict(X_test)
# Making the Confucion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

# Makeing classification report
from sklearn.metrics import classification_report
target_names = ['CN', 'EMCI', 'LMCI','AD','SMC']
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

# pocet stromov, number of features max depth, pocet estimatorov, learnig rate
        
        
        
