# %%
#%pip install imbalanced-learn

# %%
import os.path
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

subjects = [102, 104, 105, 107, 109, 110, 111, 115, 116, 117, 118, 120, 126, 127, 130, 131, 132, 133, 135, 138, 141, 143, 144]
col = ['1','2','3','Label', 'Frontal P3 mean', 'Frontal P3 STD', 'Posterior P3 mean', 'Posterior P3 STD', 'Frontal alpha mean', 
           'Posterior alpha mean', 'Alpha variability', 'Reaction time Mean', 'Reaction time variability', 'Accuracy', 'Frontal P3 log energy entropy', 
           'Frontal P3 Shannon entropy', 'Frontal P3 SURE entropy', 'Frontal P3 Skewness', 'Frontal P3 Kurtosis', 'Frontal alpha log energy entropy',
           'Frontal alpha Shannon entropy', 'Frontal alpha SURE entropy', 'Frontal alpha Skewness', 'Frontal alpha Kurtosis', 
           'Posterior P3 log energy entropy', 'Posterior P3 Shannon entropy', 'Posterior P3 SURE entropy', 'Posterior P3 Skewness', 'Posterior P3 Kurtosis', 
           'Posterior alpha log energy entropy', 'Posterior alpha Shannon entropy', 'Posterior alpha SURE entropy', 'Posterior alpha Skewness',
           'Posterior alpha Kurtosis'
]

#Load all subject mat files, append TR and TUR structures to dataframe
for a in subjects:
    file = 'Feature_data_'+str(a)
    loc = os.path.join('C:/Users/pisis/OneDrive - University of Calgary/2024/AIRS/TR and TUT data',file)
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
cleanData = totalDF.iloc[:,3:]
# print(cleanData)
# print(cleanData.Label.unique())

# %%
from imblearn.over_sampling import SMOTE


