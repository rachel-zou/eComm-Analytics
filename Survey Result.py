##########################  
# Import libraries
##########################
import pandas as pd
import pyodbc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor 

##########################  
# Connect to Teradata and read survey data
##########################
from util import credentials as cr
USER = cr.user_credentials()["user"]
PWDTERA = cr.user_credentials()["password_tera"]

conn_tera = pyodbc.connect(r'DSN=PHTDPRD;UID='+USER+';PWD='+PWDTERA+'')
cursor_tera = conn_tera.cursor()

df = pd.read_sql('''
LOCKING ROW ACCESS
select a.*, cast(cast(left(timestamp_ind,19) as timestamp) - interval '8' hour as date) as survey_dte 
from temp_marketing.apptentive_survey_stg1 a              
'''
, conn_tera)

df = df.drop_duplicates()
##########################  
# Data Cleaning
##########################
# Check statistics of the data
def knowingData(data):
    col_type = data.dtypes
    total = data.isnull().count()
    missing = data.isnull().sum()
    missing_per = round(data.isnull().sum() / data.isnull().count() * 100, 2)
    unique_values = data.nunique()
 
    max_value = data.max()
    min_value = data.min()
    median_value = data.median()
    
    result = []
    for column in data.columns:
        top3 = round(data[column].value_counts().iloc[:3] / data[column].value_counts().sum(),2).to_dict()
        result.append(sorted(top3.items(), key=lambda kv: kv[1], reverse=True))
    result = pd.Series(result,index=data.columns)

    df = pd.concat([col_type, total, missing, missing_per, unique_values, max_value, min_value, median_value, result], axis=1, 
                   keys=['Type','Total','Missing','Percent','Unique','Max','Min','Median','Top3 (Value and %)'], sort=True)     
    df.sort_values(by=['Type','Unique','Percent'],ascending=[1,1,0],inplace=True)
    
    return df

stats = knowingData(df)

''' 
By looking at the stats, we can see last_nm and email are all null, drop them. 
Also, since we don't plan to predict future survey scores, 
but just trying to understand the drivers for satisfaction, 
drop columns relevant to some generic identifiers, 
such as timestamp_ind, HHID, storenumber, CustomerID, survey_dte
Drop os_api_level too since it's only available for Android phones,
and it looks like it's corresponding to column os_version.   
Thus I will keep os_version but drop os_api_level. 
'''
df = df.drop(['timestamp_ind', 'email', 'Last_nm', 'HHID', 'storenumber', 'CustomerID','survey_dte','os_api_level'], 1)

# For survey questions, if the customer select the description for that question, map it to 1. Blank answers map to 0.
q_list = [c for c in df.columns if c.startswith('Q') and c not in ['Q1','Q15']]

for q in q_list:
    df[q] = df[q].apply(lambda x: (len(x)>0)*1)
    
# Fill in missing values for apple devices  
df['manufacturer'] = np.where(df['model'].str.startswith(('iPhone','iPad','"iPad','iPod')), 'Apple', df['manufacturer'])

# Fill in missing values using majority
df['manufacturer'] = np.where(df['manufacturer']=='', 'Apple', df['manufacturer'])  
df['locale_country_code'] = np.where(df['locale_country_code']=='', 'US', df['locale_country_code'])
df['locale_language_code'] = np.where(df['locale_language_code']=='', 'en', df['locale_language_code'])
df['os_name'] = np.where(df['os_name']=='', 'iOS', df['os_name'])

# Null carrier may mean wireless connection, fill in using NA
df['carrier'] = np.where(df['carrier']=='', 'NA', df['carrier'])

# Simplify versions
df['os_version'] = df['os_version'].apply(lambda x: x.split('.')[0]) 
df['app_version'] = df['app_version'].apply(lambda x: x[0:3])

# Drop model for now since it's too granular. manufactuer looks good enough for me
df.drop(['model'],1,inplace=True)

# Group smaller buckets in locale_country_code into "Others" category
locale_country_code_other = np.setdiff1d(df['locale_country_code'].unique().tolist(),['US','CA','GB','CA'])
df['locale_country_code'] = np.where(df['locale_country_code'].isin(locale_country_code_other), 'Others', df['locale_country_code']) 

# Group smaller buckets in manufacturer into "Others" category
df['manufacturer'] = np.where(df['manufacturer']=='google', 'Google', df['manufacturer'])
df['manufacturer'] = np.where(df['manufacturer']=='Huawei', 'HUAWEI', df['manufacturer'])

manufacturer_other = np.setdiff1d(df['manufacturer'].unique().tolist(),['Apple','samsung','LGE','motorola','Google','OnePlus','ZTE'])
df['manufacturer'] = np.where(df['manufacturer'].isin(manufacturer_other), 'Others', df['manufacturer']) 

# Group smaller buckets in carrier into "Others" category
carrier_other = df['carrier'].value_counts()[df['carrier'].value_counts()<10].index.tolist()
df['carrier'] = np.where(df['carrier'].isin(carrier_other), 'Others', df['carrier'])

# Convert Q1 Q15 to integers
df['Q1'] = df['Q1'].astype(int)
df['Q15'] = df['Q15'].astype(int)

# reshuffle dataframes
df = df.sample(frac=1)

df_backup = df.copy(deep=True)
##########################  
# Regression Model - Predict Q1 as a continuous score
##########################
# Convert categorical features
cat_cols = ['app_version',  'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9',
       'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'os_name', 'os_version',
       'manufacturer', 'locale_country_code', 'locale_language_code',
       'carrier']

X = df.drop(['Q1'],1)
y = df['Q1']

for col in cat_cols:
    dummies = pd.get_dummies(X[col]).rename(columns=lambda x: col + '.' + str(x))
    X = pd.concat([X, dummies], axis=1)
    X.drop([col], inplace=True, axis=1)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Will try LR and XGBoost. Define plotting functions
scaler = MinMaxScaler()
lr = LinearRegression()
xgb = XGBRegressor(objective ='reg:squarederror')

def plotCoefficients(model, X_train):
    if model == xgb:
        coefs = pd.DataFrame(model.feature_importances_, X_train.columns)
    else:
        coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    plt.tight_layout()
    
    plt.show()

# Training LR model
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
        
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

lr.fit(X_train_scaled, y_train)
  
y_pred = lr.predict(X_test_scaled)
r2_score(y_test, y_pred)           
mean_squared_error(y_test, y_pred) 

plotCoefficients(lr, X_train)

# Training xgb model
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_pred[y_pred < 0] = 0
r2_score(y_test, y_pred)            
mean_squared_error(y_test, y_pred)  

plotCoefficients(xgb, X_train)

# Only plot the top 20 features (in a vertical bar chart)
coefs = pd.DataFrame(xgb.feature_importances_, X_train.columns)
coefs.columns = ["coef"]
coefs["abs"] = coefs.coef.apply(np.abs)
coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

plt.figure(figsize=(7, 15))   
sns.barplot(x = coefs[:20].coef, y = coefs[:20].index, alpha = 0.8, color='blue')
plt.title("Top 20 Feature Importance for Survey Result")

##########################  
# Classification Model (Random Forest) - Predict Q1 as 5 classes
##########################
df = df_backup.copy(deep=True)

# Creating the dependent variable class
factor = pd.factorize(df['Q1'])
df.Q1 = factor[0]
definitions = factor[1]

X = df.drop(['Q1'],1)
y = df['Q1']

for col in cat_cols:
    dummies = pd.get_dummies(X[col]).rename(columns=lambda x: col + '.' + str(x))
    X = pd.concat([X, dummies], axis=1)
    X.drop([col], inplace=True, axis=1)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_test_pred = classifier.predict(X_test)
# Reverse factorize 
reversefactor = dict(zip(range(5),definitions))

y_test = np.vectorize(reversefactor.get)(y_test)
y_test_pred = np.vectorize(reversefactor.get)(y_test_pred)
print(pd.crosstab(y_test, y_test_pred, rownames=['Actual Q1 Score'], colnames=['Predicted Q1 Score']))
#cm = confusion_matrix(y_test, y_test_pred)

accuracy_score(y_test, y_test_pred)   
##########################  
# Classification Model (Random Forest) - Predict Q1 as 3 classes (Group 1~2 into 1~2, 3 into 3, 4~5 into 4~5)
##########################
df = df_backup.copy(deep=True)

df['Q1'] = df['Q1'].apply(lambda x: '1~2' if x in [1,2] else '3' if x ==3 else '4~5')

factor = pd.factorize(df['Q1'])
df.Q1 = factor[0]
definitions = factor[1]

X = df.drop(['Q1'],1)
y = df['Q1']

for col in cat_cols:
    dummies = pd.get_dummies(X[col]).rename(columns=lambda x: col + '.' + str(x))
    X = pd.concat([X, dummies], axis=1)
    X.drop([col], inplace=True, axis=1)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_test_pred = classifier.predict(X_test)
# Reverse factorize 
reversefactor = dict(zip(range(5),definitions))

y_test = np.vectorize(reversefactor.get)(y_test)
y_test_pred = np.vectorize(reversefactor.get)(y_test_pred)
print(pd.crosstab(y_test, y_test_pred, rownames=['Actual Q1 Score'], colnames=['Predicted Q1 Score']))

#cm = confusion_matrix(y_test, y_test_pred)
accuracy_score(y_test, y_test_pred)   

#feature_importances = pd.DataFrame(classifier.feature_importances_,
#                                   index = X_train.columns,
#                                   columns=['importance']).sort_values('importance', ascending=False)

coefs = pd.DataFrame(classifier.feature_importances_, X_train.columns)
coefs.columns = ["coef"]
coefs["abs"] = coefs.coef.apply(np.abs)
coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

plt.figure(figsize=(7, 15))   
sns.barplot(x = coefs[:20].coef, y = coefs[:20].index, alpha = 0.8, color='blue')
plt.title("Top 20 Feature Importance for Survey Result")

##########################  
# Classification Model (Random Forest) - Hyperparameter tuning 
##########################
## Add grid search
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 8)]
# Number of features to consider at every split
max_features = ['auto', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
## Use the random grid to search for best hyperparameters
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, and use all available cores (n_jobs = -1). 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 40, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_

'''
{'n_estimators': 200,
 'min_samples_split': 2,
 'min_samples_leaf': 4,
 'max_features': 'auto',
 'max_depth': 10,
 'bootstrap': True}
'''

base_model = RandomForestClassifier(**rf_random.best_params_)
base_model.fit(X_train, y_train)

y_test_pred = base_model.predict(X_test)
reversefactor = dict(zip(range(5),definitions))

y_test = np.vectorize(reversefactor.get)(y_test)
y_test_pred = np.vectorize(reversefactor.get)(y_test_pred)
print(pd.crosstab(y_test, y_test_pred, rownames=['Actual Q1 Score'], colnames=['Predicted Q1 Score']))

#cm = confusion_matrix(y_test, y_test_pred)
base_accuracy = accuracy_score(y_test, y_test_pred)  

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 15, None],
    'max_features': ['auto', 2, 3],
    'min_samples_leaf': [3,4,5],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [100, 200, 300, 400]
}

rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = None, verbose = 2, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)  # will take a long time here
grid_search.best_params_

'''
{'bootstrap': True,
 'max_depth': None,
 'max_features': 'auto',
 'min_samples_leaf': 5,
 'min_samples_split': 3,
 'n_estimators': 100}
'''

df = df_backup.copy(deep=True)

#df['Q1'] = df['Q1'].apply(lambda x: '1~2' if x in [1,2] else '3' if x ==3 else '4~5')

factor = pd.factorize(df['Q1'])
df.Q1 = factor[0]
definitions = factor[1]
print(df.Q1.head())
print(definitions)

X = df.drop(['Q1'],1)
y = df['Q1']

for col in cat_cols:
    dummies = pd.get_dummies(X[col]).rename(columns=lambda x: col + '.' + str(x))
    X = pd.concat([X, dummies], axis=1)
    X.drop([col], inplace=True, axis=1)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)
reversefactor = dict(zip(range(5),definitions))

y_test = np.vectorize(reversefactor.get)(y_test)
y_test_pred = np.vectorize(reversefactor.get)(y_test_pred)
print(pd.crosstab(y_test, y_test_pred, rownames=['Actual Q1 Score'], colnames=['Predicted Q1 Score']))

#cm = confusion_matrix(y_test, y_test_pred)
grid_accuracy = accuracy_score(y_test, y_test_pred) 

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy)) 