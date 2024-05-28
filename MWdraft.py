# %%
#Install list of libraries
# %pip install imbalanced-learn
# %pip install numpy
# %pip install pandas
# %pip install matplotlib
# %pip install scikit-learn
# %pip install scipy
# %pip install seaborn --upgrade
# %pip install graphviz

# %%
#In case you want to reload the modules automatically (imports aren't cached properly)
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Setup

# %%
from os import path, getcwd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# Tree Visualisation
from sklearn.tree import export_graphviz
# from IPython.display import Image
import graphviz

subjects = [102, 104, 105, 107, 110, 111, 115, 116, 117, 118, 120, 126, 127, 130, 131, 132, 133, 135, 138, 141, 143, 144]
col = ['1','2','3','Label', 'Frontal P3 mean', 'Frontal P3 STD', 'Posterior P3 mean', 'Posterior P3 STD', 'Frontal alpha mean', 
           'Posterior alpha mean', 'Alpha variability', 'Reaction time Mean', 'Reaction time variability', 'Accuracy', 'Frontal P3 log energy entropy', 
           'Frontal P3 Shannon entropy', 'Frontal P3 SURE entropy', 'Frontal P3 Skewness', 'Frontal P3 Kurtosis', 'Frontal alpha log energy entropy',
           'Frontal alpha Shannon entropy', 'Frontal alpha SURE entropy', 'Frontal alpha Skewness', 'Frontal alpha Kurtosis', 
           'Posterior P3 log energy entropy', 'Posterior P3 Shannon entropy', 'Posterior P3 SURE entropy', 'Posterior P3 Skewness', 'Posterior P3 Kurtosis', 
           'Posterior alpha log energy entropy', 'Posterior alpha Shannon entropy', 'Posterior alpha SURE entropy', 'Posterior alpha Skewness',
           'Posterior alpha Kurtosis'
]
cwd = getcwd()
target_names = ['Task Unrelated Thought', 'Task Related Thought']

# %% [markdown]
# # Data Collection
# Loading mat files into dataframe and removing unnecessary columns. NaN values are replaced with 0. 
# 
# Should any extra subjects be re-included in the future, their .mat file should be copied into 'AIRES_project/TR and TUT data', and their subject number added to the list of subjects.

# %%
#Load all subject mat files, append TR and TUR structures to dataframe
for a in subjects:
    file = 'Feature_data_'+str(a)+'.mat'
    #Absolute path to mat file:
    #loc = os.path.join('C:/Users/pisis/OneDrive - University of Calgary/2024/AIRS/TR and TUT data',file)
    loc = path.join(cwd, 'TR and TUT data', file)
    subData = loadmat(loc)['data']
    subData_TR = subData['TR'][0,0]
    subData_TUR = subData['TUR'][0,0]
    subDF_TR = pd.DataFrame(subData_TR, columns = col)
    subDF_TUR = pd.DataFrame(subData_TUR, columns = col)
    if a==subjects[0]:
        totalDF = pd.concat([subDF_TR,subDF_TUR])
    else:
        totalDF = pd.concat([totalDF, subDF_TR])
        totalDF = pd.concat([totalDF, subDF_TUR])

#Show Data with NaN values:
# print(totalDF[totalDF.isnull().any(axis=1)])
# NOTE: Subject 109 has NaN values in the Reaction time Mean and Reaction time variability columns. Excluded from analysis.
# totalDF.fillna(0, inplace=True)

totalDF.reset_index(drop=True, inplace=True)
print(totalDF)

target_Data = totalDF.iloc[:,3:]
# print(targetData)
X = target_Data.iloc[:,1:]
Y = target_Data.Label
Y = Y - 1
# Verify that Labels contain only 1 and 2:
# print(targetData.Label.unique())



# %% [markdown]
# # PreProcessing: Split and Normalize
# 
# Using the built-in normalize, alternatives: StandardScaler, MinMaxScaler

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


normalized_X_train = normalize(X_train)


# %% [markdown]
# # SMOTE Data Augmentation
# Data is oversampled to correct for inbalances.

# %%
smote = SMOTE() #sampling_strategy='minority' ?
X_smote, y_smote = smote.fit_resample(normalized_X_train, y_train)
print("Before oversampling: ", y_train.value_counts())
print("After oversampling: ", y_smote.value_counts())

# %% [markdown]
# # Logistic Regression
# Binary classification that uses the sigmoid function. Prone to overfitting.
# 
# *Learned many of the techniques below from __[DataCamp](https://www.datacamp.com/tutorial/understanding-logistic-regression-python)__*.

# %%
logreg = LogisticRegression(random_state = 32, max_iter=1100, solver='lbfgs')
logreg.fit(X_train, y_train)
log_Y_Pred = logreg.predict(X_test)

# %% [markdown]
# 

# %%
cnf_matrix = metrics.confusion_matrix(y_test, log_Y_Pred)
sensitivity_score = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
specificity_score = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
mcc = metrics.matthews_corrcoef(y_test, log_Y_Pred)
accuracy = metrics.accuracy_score(y_test, log_Y_Pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_test, log_Y_Pred)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(cnf_matrix)
print("Sensitivity: ", sensitivity_score)
print("Specificity: ", specificity_score)
print("Matthews Correlation Coefficient: ", mcc)
print("AUC: ", auc)
print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print(metrics.classification_report(y_test, log_Y_Pred, target_names=target_names))

# %%
class_names = ['TR', 'TUR'] #['Task Related Thought', 'Task Unrelated Thought']
plt.figure(figsize=(5, 5))
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)

sns.heatmap(cnf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=class_names, yticklabels=class_names)

# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label') 
plt.xlabel('Predicted label', loc='center')
plt.show()

# %%
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# %% [markdown]
# # Random Forest
# Multiple decision trees are created using random subsets of data and features. The most popular result of all these trees becomes the prediction.

# %%
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
RaFo_Y_Pred = rf.predict(X_test)

# %%
cnf_matrix = metrics.confusion_matrix(y_test, RaFo_Y_Pred)
sensitivity_score = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
specificity_score = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
mcc = metrics.matthews_corrcoef(y_test, RaFo_Y_Pred)
accuracy = metrics.accuracy_score(y_test, RaFo_Y_Pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_test, RaFo_Y_Pred)
auc = metrics.roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(cnf_matrix)
print("Sensitivity: ", sensitivity_score)
print("Specificity: ", specificity_score)
print("Matthews Correlation Coefficient: ", mcc)
print("AUC: ", auc)
print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print(metrics.classification_report(y_test, RaFo_Y_Pred, target_names=target_names))

# %%
# From https://www.datacamp.com/tutorial/random-forests-classifier-python

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=2, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     display(graph)

# %%
class_names = ['TR', 'TUR'] #['Task Related Thought', 'Task Unrelated Thought']
plt.figure(figsize=(5, 5))
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)

sns.heatmap(cnf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=class_names, yticklabels=class_names)

# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label') 
plt.xlabel('Predicted label', loc='center')
plt.show()

# %% [markdown]
# # K-Nearest Neighbours

# %%
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
KNN_y_predict = model.predict(X_test)

# %%
cnf_matrix = metrics.confusion_matrix(y_test, KNN_y_predict)
sensitivity_score = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
specificity_score = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
mcc = metrics.matthews_corrcoef(y_test, KNN_y_predict)
accuracy = metrics.accuracy_score(y_test, KNN_y_predict)
balanced_accuracy = metrics.balanced_accuracy_score(y_test, KNN_y_predict)
auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
print(cnf_matrix)
print("Sensitivity: ", sensitivity_score)
print("Specificity: ", specificity_score)
print("Matthews Correlation Coefficient: ", mcc)
print("AUC: ", auc)
print("Accuracy: ", accuracy)
print("Balanced Accuracy: ", balanced_accuracy)
print(metrics.classification_report(y_test, KNN_y_predict, target_names=target_names))


