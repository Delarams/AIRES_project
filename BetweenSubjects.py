# %% [markdown]
# # Setup

# %% [markdown]
# ## Easy Install

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
# %pip install xgboost
# %pip install lightgbm
# %pip install catboost

# %% [markdown]
# ## Imports and Variables

# %%
from os import path, getcwd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif, SelectKBest, SelectFromModel
from scipy.stats import kendalltau

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
results_file = 'Results.xlsx'

# Fill this with the models you would like to test:
# regressors = [LogisticRegression(max_iter=1800, random_state=42), RandomForestClassifier(random_state=42), GradientBoostingClassifier(random_state=42), SVC(random_state=42), KNeighborsClassifier(), 
            #   XGBClassifier(random_state=42), LGBMClassifier(random_state=42), CatBoostClassifier(random_state=42)]
regressors = [CatBoostClassifier(depth= 6, iterations= 300, learning_rate= 0.1, random_state=42), 
              RandomForestClassifier(bootstrap= False, max_depth= 29, max_features= "log2", min_samples_leaf= 1, 
                                     min_samples_split= 2, n_estimators= 200, random_state=42),]

# %% [markdown]
# Alpha variability and all with posterior features --> see performance,  
# Then include behavioural features: 'Reaction time Mean', 'Reaction time variability',and see how it performs
# 
# Feature ranking algorithms (top 5, top 10 features)
# correlation matrix between features to identify important ** try first

# %% [markdown]
# ## Data Collection

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
# print(totalDF)


# %% [markdown]
# ## Split and Normalize
# There are some [different normalization techniques](https://www.geeksforgeeks.org/ml-feature-scaling-part-2/), but all seem to give the same result in our models.

# %%
Y = totalDF.Label
Y = Y - 1
all_features = totalDF.iloc[:, 4:]

# %% [markdown]
# ## Feature Selection
# Feature selection is for determining the most important features in our data. We compared all features, the alpha + posterior features, and finally, the alpha + posterior + behavioural features before using the built-in feature analysis (suggested by Sarah).  
# [There are many types of feature selection techniques](https://youtu.be/LTE7YbRexl8?si=xW9kJt1lciKEKwAW). 
# 1. Filter-based techniques:
#     - Correlation
#     - Variance threshold
#     - Chi squared
#     - Anova
#     - Information Gain
# 2. Wrapper techniques:
#     - Recursive Feature Elimination (RFE)
# 3. Embed techniques
#     - L1 & L2
#     - Pruning/Decision trees

# %%
ap_columns = ['Posterior P3 mean', 'Posterior P3 STD', 'Posterior alpha mean', 'Alpha variability', 'Accuracy', 'Posterior P3 log energy entropy', 
              'Posterior P3 Shannon entropy', 'Posterior P3 SURE entropy', 'Posterior P3 Skewness', 'Posterior P3 Kurtosis', 
              'Posterior alpha log energy entropy', 'Posterior alpha Shannon entropy', 'Posterior alpha SURE entropy', 'Posterior alpha Skewness',
              'Posterior alpha Kurtosis']
ap_features = totalDF[ap_columns]

apb_columns = ['Posterior P3 mean', 'Posterior P3 STD', 'Posterior alpha mean', 'Alpha variability', 'Reaction time Mean', 'Reaction time variability', 'Accuracy',
               'Posterior P3 log energy entropy', 'Posterior P3 Shannon entropy', 'Posterior P3 SURE entropy', 'Posterior P3 Skewness', 'Posterior P3 Kurtosis', 
               'Posterior alpha log energy entropy', 'Posterior alpha Shannon entropy', 'Posterior alpha SURE entropy', 'Posterior alpha Skewness',
               'Posterior alpha Kurtosis']
apb_features = totalDF[apb_columns]

# %% [markdown]
# ### Correlation Matrix Feature Selection

# %%
# Correlation Matrix:
corrMat = all_features.corr()
plt.figure(figsize=(20,20))
# sns.heatmap(corrMat, annot=True, cmap='Blues', fmt=".2f")

# All features with correlation ge than .80:
columns_to_drop = ['Frontal P3 log energy entropy','Frontal alpha log energy entropy', 'Frontal alpha Kurtosis', 
                   'Posterior P3 log energy entropy', 'Posterior alpha log energy entropy', 'Posterior alpha Kurtosis']
uncorr_features = all_features.drop(columns=columns_to_drop, axis=1)
# sns.heatmap(uncorr_features.corr(), annot=True, cmap='Blues', fmt=".2f")

# %% [markdown]
# ### Kendall's Tau Correlation Matrix

# %%
kenmat = all_features.corr(method='kendall')
# sns.heatmap(kenmat, annot=True, cmap='Blues', fmt=".2f")

# All features with kendall correlation greater than .80
columns_to_drop = ['Frontal P3 log energy entropy', 'Posterior P3 log energy entropy']
kendall_features = all_features.drop(columns=columns_to_drop, axis=1)

# %% [markdown]
# ### Variance Threshold

# %%
vt = VarianceThreshold(threshold=0.1)
vt.fit(all_features)
mask = vt.get_support()
print("Features excluded: ", all_features.columns[~mask].values)
vt_features = all_features.loc[:, mask]

# %% [markdown]
# ### SelectKBest - ANOVA

# %%
test = SelectKBest(score_func=f_classif, k=30)
fit = test.fit(all_features, Y)
mask = fit.get_support()
print("Features excluded: ", all_features.columns[~mask].values)
anova_features = all_features.loc[:, mask]

# %% [markdown]
# ### Final Feature Selection

# %%
# 'AP' = alpha + posterior, 'APB' = alpha + posterior + behvaioural, 'all' = all features
Select_features = 'uncorr'
Notes = ''

if Select_features == 'AP':
    X = ap_features
elif Select_features == 'APB':
    X = apb_features
elif Select_features == 'uncorr':
    X = uncorr_features
elif Select_features == 'vt':
    X = vt_features
elif Select_features == 'ANOVA':
    X = anova_features
elif Select_features == 'kendalls':
    X = kendall_features
else: 
    X = all_features
    


# print(X.columns)
# Verify that Labels contain only 0 and 1:
# print(X.Label.unique())

# %% [markdown]
# # K-Fold Cross Validation

# %% [markdown]
# ## Variables init
# Choose what type of KFold, which measures you would like, what info you would like to store in results. Some resources:
# - [svm](https://www.youtube.com/watch?v=efR1C6CvhmE)
# - [Gradient Boosting info](https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/)

# %%
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
index = []
scores = {"Accuracy": [], "BA": [], "Matt_Corr_Coef": [], "Cnf_Matr": None, "AUC": []}
results = {'Timestamp': [], 'Features': Select_features, 'CrossVal': type(kf).__name__, 'model': [], "Accuracy": [], "BA": [], "Matt_Corr_Coef": [], 'AUC': [], 'CnfM00': [], 'CnfM01': [], 
           'CnfM10': [], 'CnfM11': [], 'Notes': Notes}
scoring = ['accuracy', 'balanced_accuracy', 'matthews_corrcoef']   


# %%
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    sm = SMOTE(random_state=42)
    X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)
    
    normalized = 3
    if normalized == 1:
        scaler = StandardScaler()
    elif normalized == 2:
        scaler = MinMaxScaler()
    elif normalized == 3:
        scaler = Normalizer()
    X_train = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
    
    

# %%
for regressor in regressors:
    for train_index, test_index in kf.split(X, Y):
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = Y.iloc[train_index], Y.iloc[test_index]

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train_kf, y_train_kf)
        
        normalized = 3
        
        sc = StandardScaler()
        mms = MinMaxScaler()
        ns = Normalizer()
        if normalized == 1:
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test_kf)
        elif normalized == 2:
            X_train = mms.fit_transform(X_train)
            X_test = mms.transform(X_test_kf)
        elif normalized == 3:
            X_train = ns.fit_transform(X_train)
            X_test = ns.transform(X_test_kf)
        # X_train = pd.DataFrame(X_train_res, columns=all_features.columns)
        # X_test = pd.DataFrame(X_test_kf, columns=all_features.columns)
                
        cv_results = cross_validate(regressor, X_train, y_train, cv=5, scoring=scoring)
        scores["Accuracy"].append(cv_results['test_accuracy'].mean())
        scores["BA"].append(cv_results['test_balanced_accuracy'].mean())
        scores["Matt_Corr_Coef"].append(cv_results['test_matthews_corrcoef'].mean())
        cnf_matrix = metrics.confusion_matrix(y_test_kf, regressor.fit(X_train, y_train).predict(X_test))
        if scores["Cnf_Matr"] is None:
            scores["Cnf_Matr"] = cnf_matrix
        else:
            scores["Cnf_Matr"] = np.mean(np.array([scores['Cnf_Matr'], cnf_matrix]), axis=0 )
        scores["AUC"].append(metrics.roc_auc_score(y_test_kf, regressor.fit(X_train, y_train).predict(X_test)))
        
        
    print('\n')
    print(type(regressor).__name__)
    results['model'].append(type(regressor).__name__)
    
    for key in scores:
        if key != "Cnf_Matr":
            print(key, ":", np.mean(scores[key]))
            results[key].append(np.mean(scores[key]))
    print("Cnf_Matr: \n", scores["Cnf_Matr"])
    print("\n")    
    
    results['CnfM00'].append(scores["Cnf_Matr"][0][0])
    results['CnfM01'].append(scores["Cnf_Matr"][0][1])
    results['CnfM10'].append(scores["Cnf_Matr"][1][0])
    results['CnfM11'].append(scores["Cnf_Matr"][1][1])
    results['Timestamp'].append(pd.Timestamp.now())
    
    scores = {"Accuracy": [], "BA": [], "Matt_Corr_Coef": [], "Cnf_Matr": None, "AUC": []}
        

# %% [markdown]
# ## Record Results
# Add these Results and Test Conditions to Results.xlsx. Will delete duplicate results within conditions.

# %%
results['Normalized'] = normalized
df_newScores = pd.DataFrame(results)
print(df_newScores)
df_existingRecord = pd.read_excel(results_file)
df_combined = pd.concat([df_existingRecord, df_newScores], ignore_index=True)
df_combined.drop_duplicates(subset=['Features', 'model', "Accuracy", "BA", "Matt_Corr_Coef", 'AUC', 'CnfM00', 'CnfM01', 'CnfM10', 'CnfM11', 'Notes'], keep='last', inplace = True)
df_combined.sort_values(by='Matt_Corr_Coef', ascending=False, inplace=True)
df_combined.to_excel(results_file, index=False)

# %% [markdown]
# # Tuning Hyperparameters

# %%
randfor_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [17, 19, 21],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'bootstrap': [False]
}
random_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                             param_grid=randfor_param_grid, cv=10, verbose=2,
                             n_jobs=-1, scoring='balanced_accuracy')

# for train_index, test_index in kf.split(X, Y):
#     X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
#     y_train_kf, y_test_kf = Y.iloc[train_index], Y.iloc[test_index]

#     sm = SMOTE(random_state=42)
#     X_train, y_train = sm.fit_resample(X_train_kf, y_train_kf)
    
#     normalized = 3
#     ns = Normalizer()
#     X_train = ns.fit_transform(X_train)
#     X_test = ns.transform(X_test_kf)
    
#     random_search.fit(X_train, y_train)
#     print(random_search.best_params_)
    
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
normalized = 3
ns = Normalizer()
X_train = ns.fit_transform(X_train)
X_test = ns.transform(X_test)
random_search.fit(X_train, y_train)
print(random_search.best_params_)


# %%
print(f'Train Accuracy: {random_search.best_score_}')
print(f'Test Accuracy: {random_search.score(X_test, y_test)}')

# %% [markdown]
# ## CatBoost
# [Hyperparameter tutorial](https://www.geeksforgeeks.org/catboost-parameters-and-hyperparameters/)

# %%
X = uncorr_features
catboost_pool = Pool(X, label = Y)
params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'MultiClass',
    'verbose': 200,
    'random_state': 42
}
cv_results, cv_model = cv(catboost_pool, params, fold_count=5, plot=True, verbose=200, return_models=True, stratified=True)
available_metrics = [metric for metric in cv_results.columns if 'test' in metric]
print(available_metrics)

# %%
# Define the parameter grid
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.001],
    'depth': [4, 6, 8]
}

# Create the CatBoostClassifier
model = CatBoostClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Create a new CatBoostClassifier with the best parameters
# best_model = CatBoostClassifier(**best_params)

# %%
print(best_params)
print(best_score)


