# %%
from os import path, getcwd
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

subjects = [102, 104, 105, 107, 109, 110, 111, 115, 116, 117, 118, 120, 126, 127, 130, 131, 132, 133, 135, 138, 141, 143, 144]
col = ['1','2','3','Label', 'Frontal P3 mean', 'Frontal P3 STD', 'Posterior P3 mean', 'Posterior P3 STD', 'Frontal alpha mean', 
           'Posterior alpha mean', 'Alpha variability', 'Reaction time Mean', 'Reaction time variability', 'Accuracy', 'Frontal P3 log energy entropy', 
           'Frontal P3 Shannon entropy', 'Frontal P3 SURE entropy', 'Frontal P3 Skewness', 'Frontal P3 Kurtosis', 'Frontal alpha log energy entropy',
           'Frontal alpha Shannon entropy', 'Frontal alpha SURE entropy', 'Frontal alpha Skewness', 'Frontal alpha Kurtosis', 
           'Posterior P3 log energy entropy', 'Posterior P3 Shannon entropy', 'Posterior P3 SURE entropy', 'Posterior P3 Skewness', 'Posterior P3 Kurtosis', 
           'Posterior alpha log energy entropy', 'Posterior alpha Shannon entropy', 'Posterior alpha SURE entropy', 'Posterior alpha Skewness',
           'Posterior alpha Kurtosis'
]
cwd = getcwd()

# %%
#Load all subject mat files, append TR and TUR structures to dataframe
for a in subjects:
    file = 'Feature_data_'+str(a)+'.mat'
    #Absolute path to mat file:
    #loc = os.path.join('C:/Users/pisis/OneDrive - University of Calgary/2024/AIRS/TR and TUT data',file)
    loc = path.join(cwd, 'TR and TUT data', file)
    subData = loadmat(loc)['data']
    subDataTR = subData['TR'][0,0]
    subDataTUR = subData['TUR'][0,0]
    subDFTR = pd.DataFrame(subDataTR, columns = col)
    subDFTUR = pd.DataFrame(subDataTUR, columns = col)
    if a==subjects[0]:
        totalDF = pd.concat([subDFTR,subDFTUR])
    else:
        totalDF = pd.concat([totalDF, subDFTR])
        totalDF = pd.concat([totalDF, subDFTUR])
print(totalDF)


# %%
#Show Data with NaN values:
# print(totalDF[totalDF.isnull().any(axis=1)])
# NOTE: Subject 109 has NaN values in the Reaction time Mean and Reaction time variability columns. Will fill with 0-values, may need to exclude
totalDF.fillna(0, inplace=True)
cleanData = totalDF.iloc[:,3:]
X = cleanData.iloc[:,1:]
Y = cleanData.Label
# print(cleanData)
# print(cleanData.Label.unique())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# %%
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)

# %%
smote = SMOTE() #sampling_strategy='minority' ?
X_smote, y_smote = smote.fit_resample(X_train, y_train)


