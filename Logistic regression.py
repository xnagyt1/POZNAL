
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

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
X = X.replace('-4', np.nan)
X = X.dropna(axis=0, how='any')
X_full = X

Y = X.iloc[:,16]
X = X.loc[:, X.columns != 'DX_bl']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

corelogramTests = X_full.iloc[:,[1,2,3,4,15,16]]
corelogramAgeGenderEducationMaried = X_full.iloc[:,[15,17,18,19,16]]
sns.pairplot(corelogramAgeGenderEducationMaried)
#plt.show()

le = LabelEncoder()
le.fit(X_full['DX_bl'])
X_full['DX_bl'] = le.transform(X_full['DX_bl'])


#X_full.iloc[:,16].apply(LabelEncoder().fit_transform)

#sns.pairplot(corelogramAgeGenderEducationMaried, kind="scatter", hue="DX_bl", markers=["o", "o", "o","o","o"], palette="Set2")
#.show()
 
# right: you can give other arguments with plot_kws.
#sns.pairplot(df, kind="scatter", hue="species", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
#plt.show()


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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)
"""
from imblearn.over_sampling import RandomOverSampler
X_Y_train = np.concatenate((X_train,np.reshape(Y_train,(len(Y_train),1))),axis=1)

x_0 = np.where(X_Y_train[:,33] == 0)
x_0 = X_Y_train[x_0]

x_1 = np.where(X_Y_train[:,33] == 1)
x_1 = X_Y_train[x_1]

x_2 = np.where(X_Y_train[:,33] == 2)
x_2 = X_Y_train[x_2]

x_3 = np.where(X_Y_train[:,33] == 3)
x_3 = X_Y_train[x_3]

x_4 = np.where(X_Y_train[:,33] == 4)
x_4 = X_Y_train[x_4]

ros = RandomOverSampler(random_state=0)
X_resampled, X1_resampled = ros.fit_sample(X_Y_train, x_0)
X_resampled, X2_resampled = ros.fit_sample(X_Y_train, x_1)
X_resampled, X3_resampled = ros.fit_sample(X_Y_train, x_2)
X_resampled, X4_resampled = ros.fit_sample(X_Y_train, x_3)
X_resampled, X5_resampled = ros.fit_sample(X_Y_train, x_4)"""
        

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
#Y_pred = classifier.predict_classes(X_test)
#Y_pred = (Y_pred>0.5)

# Making the Confucion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

# Makeing classification report
from sklearn.metrics import classification_report
target_names = ['AD', 'EMCI', 'LMCI','AD','SNC']
cr = classification_report(Y_test, Y_pred, target_names=target_names)
#clasification report
#feature exposion

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


        
        
        