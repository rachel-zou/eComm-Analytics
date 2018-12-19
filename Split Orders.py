'''
Explore which orders are likely to be split, therefore impacting our routing system. 
'''
import pandas as pd
import pyodbc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
##########################
from util import credentials as cr
USER = cr.user_credentials()["user"]
PWD = cr.user_credentials()["password"]

conn_sql = pyodbc.connect(r'DSN=eComm;Trusted_Connection=yes;MARS_Connection=yes')
cursor_sql = conn_sql.cursor()
##########################
### Routing information (Didn't include queries here for confidentiality)
routes = pd.read_sql('''
SELECT * FROM table
                   ''' 
                   , conn_sql)  

### Orders from last week 
ords = pd.read_sql('''
SELECT * FROM table	
                   ''' 
                   , conn_sql)  
##########################
# Split orders usually have a format of 9 digits with 01,02,03 etc at the end
# So I will remove the last two digits of these order ids to get the original order ids
routes['Order_ID2'] = routes['LocationKey'].apply(lambda x: x[0:len(x)-2] if len(x)==9 else x)

routes.sort_values(by=['Store_ID','shift','LocationKey'],ascending=[1,1,1],inplace=True) 

# An order is uniquely identified by store id and order id. Create a key for this.
routes['key'] = routes['Store_ID'].map(str) + "-" + routes['Order_ID2'].map(str)

# Some orders have different delivery date/time than their original orders. 
# Checked using the code below, looks like these orders are re-deliveries (next day, etc)
test = routes[routes.duplicated(['Order_ID2','Store_ID','OpenDateTime', 'CloseDateTime'], keep=False)==True]
test1 = routes[routes.duplicated(['Order_ID2','Store_ID'], keep=False)==True]
diff = np.setdiff1d(test1['key'].tolist(),test['key'].tolist())
del test, test1, diff

# Only consider orders with the following 4 same attributes as the split orders
split_orders = routes[routes.duplicated(['Order_ID2','Store_ID','OpenDateTime', 'CloseDateTime'], keep=False)==True]
split_orders['key'].nunique()   
split_orders['Store_ID'] = split_orders['Store_ID'].astype(int)
split_orders['Order_ID2'] = split_orders['Order_ID2'].astype(int)

# The remaining orders are not split
non_split_orders = routes[~routes['key'].isin(split_orders['key'])]  
# Exclude some irregular order ids
non_split_orders = non_split_orders[~(non_split_orders['OrderIDLen']>9)] 
non_split_orders['Store_ID'] = non_split_orders['Store_ID'].astype(int)
non_split_orders['Order_ID2'] = non_split_orders['Order_ID2'].astype(int)
 
# Now get the order details for split orders and non split orders
def order_details(df):
    df_name = pd.merge(df, ords, how = 'inner', left_on = ['Store_ID','Order_ID2'], right_on = ['ro_no','order_no'])

    df_name = df_name[['shift', 'cust_id',
       'order_no', 'ro_no', 'd_date', 'dely_pcode', 'created_dttm',
       'alloc_slot_start', 'alloc_slot_end', 'dely_window', 'slot', 'B2B',
       'UA', 'Free_Dely', 'DUG', 'bs', 'line', 'units', 'dely_surname',
       'dely_charge', 'card_typ', 'lyty_card_no', 'version_id',
       'updated_order','coupon_no']]

    df_name = df_name.drop_duplicates() 
    df_name.reset_index(inplace=True,drop=True)
    
    df_name['shift_no'] = df_name['shift'].apply(lambda x: x[1:2])
    df_name['hh_in_advance'] = df_name['alloc_slot_start']-df_name['created_dttm']
    df_name['hh_in_advance2'] = df_name['hh_in_advance'].apply(lambda x: x.total_seconds()/3600)
    return df_name     

split_orders_details = order_details(split_orders)
non_split_orders_details = order_details(non_split_orders)

##########################
# Now let's compare the attributes between split orders and non split orders
f, ax = plt.subplots(1, 1, figsize=(12, 6))
ax = sns.distplot(split_orders_details['bs'], hist=True, bins=500, color="tan", kde_kws={"shade": True}, label="split orders")
ax = sns.distplot(non_split_orders_details['bs'], hist=True, bins=500, color="b", kde_kws={"shade": True}, label="non split orders")
ax.set_xlabel('Basket Size')
ax.set_ylabel('Density')
ax.set_xlim((-100, 1500))
ax.legend()
ax.set_title('Basket Size Distribution\n', fontsize=15)

ax = sns.distplot(split_orders_details['line'], hist=True, bins=500, color="tan", kde_kws={"shade": True}, label="split orders")
ax = sns.distplot(non_split_orders_details['line'], hist=True, bins=500, color="b", kde_kws={"shade": True}, label="non split orders")
ax.set_xlabel('Lines') 
ax.set_ylabel('Density')
ax.set_xlim((-30, 160))
ax.legend()
ax.set_title('Lines Distribution\n', fontsize=15)

ax = sns.distplot(split_orders_details['units'], hist=True, bins=500, color="tan", kde_kws={"shade": True}, label="split orders")
ax = sns.distplot(non_split_orders_details['units'], hist=True, bins=500, color="b", kde_kws={"shade": True}, label="non split orders")
ax.set_xlabel('Units') 
ax.set_ylabel('Density')
ax.set_xlim((-30, 500))
ax.legend()
ax.set_title('Units Distribution\n', fontsize=15)

ax = sns.distplot(split_orders_details['hh_in_advance2'], hist=True, bins=500, color="tan", kde_kws={"shade": True}, label="split orders")
ax = sns.distplot(non_split_orders_details['hh_in_advance2'], hist=True, bins=500, color="b", kde_kws={"shade": True}, label="non split orders")
ax.set_xlabel('Hours in Advance') 
ax.set_ylabel('Density')
ax.set_xlim((-10, 150))
ax.legend()
ax.set_title('Hours in Advance Distribution\n', fontsize=15)

ax = sns.distplot(split_orders_details['coupon_no'], hist=True, color="tan", kde_kws={"shade": True}, label="split orders")
ax = sns.distplot(non_split_orders_details['coupon_no'], hist=True, color="b", kde_kws={"shade": True}, label="non split orders")
ax.set_xlabel('# Coupons Per Order') 
ax.set_ylabel('Density')
ax.set_xlim((-2, 10))
ax.legend()
ax.set_title('Coupons Distribution\n', fontsize=15)

##########################
# Treat this as a classification problem. Use a light gbm model to train the dataset and predict split orders.

# Split orders are the targets we want to identify
split_orders_details['split'] = 1
non_split_orders_details['split'] = 0

all_orders = pd.concat([split_orders_details,non_split_orders_details])

# Shuffle the whole dataset
all_orders = all_orders.sample(frac=1).reset_index(drop=True)
all_orders['dow'] = all_orders['alloc_slot_start'].apply(lambda x: x.weekday()) # Monday is 0 and Sunday is 6

# The features we are going to use in the model
feature_cols =[
        'dely_window',
        'slot', 
        'B2B', 
        'UA', 
        'Free_Dely',  
        'bs', 
        'line', 
        'units',
        'dely_charge', 
        'card_typ', 
        'version_id',
        'updated_order', 
        'coupon_no', 
        'shift_no', 
        'hh_in_advance2', 
        'dow'
        ]

# All the categorical features
cat_cols = [
       'dely_window', 'slot', 'B2B', 'UA', 'Free_Dely', 
       'card_typ', 'updated_order','shift_no', 'dow'
       ]

# Split the dataset into features and target
X = all_orders[feature_cols]
y = all_orders['split']

# Encode all the categorical columns
for col in cat_cols:
    dummies = pd.get_dummies(X[col]).rename(columns=lambda x: col + '.' + str(x))
    X = pd.concat([X, dummies], axis=1)
    X.drop([col], inplace=True, axis=1)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score, precision_score, confusion_matrix, f1_score, matthews_corrcoef
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=1) 

######## Balance Datasets - since we don't have enough positive samples here, I used over sample to increase positive targets
# Balance the 0 and 1 in training dataset - Over Sample
y_train = pd.DataFrame(y_train)
train = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)

neg = train[train['split']==0]
pos = train[train['split']==1]

pos_over = pos.sample(len(neg), replace=True)
df2 = pd.concat([neg,pos_over])
df2 = df2.sample(frac=1).reset_index(drop=True)
X_train = df2[df2.columns[0:df2.shape[1]-1]]
y_train = df2[df2.columns[-1]]
# Over Sample End

## Balance the 0 and 1 in training dataset - Under Sample
#y_train = pd.DataFrame(y_train)
#train = pd.merge(X_train, y_train, how='inner', left_index=True, right_index=True)
#
#neg = train[train['split']==0]
#pos = train[train['split']==1]
#neg = neg.sample(len(pos))
#df2 = pd.concat([neg,pos])
#df2 = df2.sample(frac=1).reset_index(drop=True)
#X_train = df2[df2.columns[0:df2.shape[1]-1]]
#y_train = df2[df2.columns[-1]]
## Under Sample End
 
############### Training models
# Define the model and parameters we will use
def run_lgb(X_train, y_train, X_valid, y_valid, X_test):
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 100,  
    'max_depth': 10,    
    'feature_fraction': 0.9,    
    'bagging_fraction': 0.95,  
    'bagging_freq': 5,          
    'learning_rate': 0.003,
    'min_data': 20 
    }
    
    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_valid, label=y_valid)
    model = lgb.train(params, lgtrain, 5000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(X_test, num_iteration=model.best_iteration)
    pred_val_y = model.predict(X_valid, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Train the model 
pred_test, churn_model, pred_val = run_lgb(X_train, y_train, X_valid, y_valid, X_test)

# Round the probabilities to get the predicted result as 1 or 0. Define the target as 1 if p > 0.5.
predicted = [round(value) for value in pred_test]
expected = y_test

# Print out metrics
print(metrics.classification_report(expected, predicted))
print(confusion_matrix(expected, predicted))                                         
print("accuracy_score: {}".format(round(accuracy_score(expected, predicted),4)))    
print("precision_score: {}".format(round(precision_score(expected, predicted),4)))  
print("recall_score: {}".format(round(recall_score(expected, predicted),4)))        
print("f1_score: {}".format(round(f1_score(expected, predicted),4)))                
print("mcc_score: {}".format(round(matthews_corrcoef(expected, predicted),4)))

print("True split order%: {}".format(y_test.sum()/len(y_test)))                
print("Predicted split order%: {}".format(predicted.count(1)/len(predicted)))  

# Using predicted value 
fpr, tpr, threshold = roc_curve(expected, predicted)
roc_auc = auc(fpr,tpr)
print("roc_auc: {}".format(roc_auc)) 

# Using probability 
fpr, tpr, threshold = roc_curve(expected, pred_test)
roc_auc = auc(fpr,tpr)
print("roc_auc: {}".format(roc_auc)) 

# Plot ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Plot feature importance
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(churn_model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("Feature Importance - Split Orders", fontsize=15)
plt.show()