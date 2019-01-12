'''
We would like to examine the retention rates for new customers 
who made their first purchases using different new customer coupons,
so we can compare which group has better retention rates and basket sizes.

The final output of this script will be 3 Cohort Curve Spreadsheets:
-By customer counts each month
-By average transaction size each month
-By order/user ration each month

Each spreadsheet will contain a separate sheet for the specific coupon.
'''
import pandas as pd
import pyodbc
import numpy as np
##########################  
# Connect to SQL and Tera
##########################
from util import credentials as cr
USER = cr.user_credentials()["user"]
PWDTERA = cr.user_credentials()["password_tera"]

conn_tera = pyodbc.connect(r'DSN=PHTDPRD;UID='+USER+';PWD='+PWDTERA+'')
cursor_tera = conn_tera.cursor()

conn_sql = pyodbc.connect(r'DSN=eComm;Trusted_Connection=yes;MARS_Connection=yes')
cursor_sql = conn_sql.cursor()

conn_sql0 = pyodbc.connect(r'DSN=atlas;Trusted_Connection=yes;MARS_Connection=yes')
cursor_sql0 = conn_sql0.cursor()
###################
# Read the file that contains all the eComm transactions 
ecomm = pd.read_csv(r'C:\Users\rzou000\Documents\Reports\Cohort\Rachel\raw_ecomm.csv')

# Query all the orders where the particular BNC coupons have been applied
promo = pd.read_sql('''
select Cust_ID, lyty_card_no, d_date, dely_surname, ro_no, order_no, CouponCode 
from TABLE
WHERE CouponCode in ('****', '****', '****')                    
                   '''
                   , conn_sql)
 
first = pd.read_sql('''
select cust_id, lyty_card_no, ro_no, order_no
from TABLE              
                   '''
                   , conn_sql)
                
first_coupon = pd.merge(promo, first, how='inner', left_on = ['ro_no','order_no'], right_on = ['ro_no','order_no'])

cpn = first_coupon['CouponCode'].unique().tolist()
###################
def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

def create_cohort(df, cpn):
    df['OrderPeriod'] = df['txn_dte'].apply(lambda x: x[0:7])
     
    df.set_index('card_nbr', inplace=True)
    df['CohortGroup'] = df.groupby(level=0)['txn_dte'].min().apply(lambda x: x[0:7])
    df['First Order Date'] = df.groupby(level=0)['txn_dte'].min().apply(lambda x: x)
    df.reset_index(inplace=True)
    
    df['First Order Date'] = pd.to_datetime(df['First Order Date'])
    # You may want to only look at new customers after a certain time    
    new = df[df['First Order Date'] > '2017-01-01']['card_nbr']
    df = df[df['card_nbr'].isin(new)]
    ########################## 
    grouped = df.groupby(['CohortGroup', 'OrderPeriod'])  
    
    cohorts = grouped.agg({'card_nbr': pd.Series.nunique,
                           'txn_dte': 'count',
                           'net_amt': np.sum})
    
    cohorts.rename(columns={'card_nbr': 'TotalUsers',
                            'txn_dte': 'TotalOrders',
                            'net_amt': 'TotalSales'}, inplace=True)
    
    cohorts['AOS'] = cohorts['TotalSales']/cohorts['TotalOrders']
    cohorts = cohorts.groupby(level=0).apply(cohort_period)
    
    # If comment out, layout will be by OrderPeriod. Otherwise, layout will be by CohortPeriod
    #cohorts.reset_index(inplace=True)
    #cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)
    
    ###################### By AOS
    cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
    cohort_group_size2 = pd.DataFrame(cohort_group_size)
    
    cohort_AOS = cohorts['AOS'].unstack(0)
    cohort_AOS2 = cohort_AOS.T
    cohort_AOS2 = pd.merge(cohort_group_size2, cohort_AOS2, how='inner', left_index=True, right_index=True)
        
    cohort_AOS2.to_excel(writer_aos, '%s' % cpn, index=True)
    ###################### By User
    user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
    user_retention2 = user_retention.T
    user_retention2 = pd.merge(cohort_group_size2, user_retention2, how='inner', left_index=True, right_index=True)
    
    user_retention2.to_excel(writer_user, '%s' % cpn, index=True)
    ###################### By Order
    user_retention_by_order = cohorts['TotalOrders'].unstack(0).divide(cohort_group_size, axis=1)
    user_retention_by_order2 = user_retention_by_order.T
    user_retention_by_order2 = pd.merge(cohort_group_size2, user_retention_by_order2, how='inner', left_index=True, right_index=True)
    
    user_retention_by_order2.to_excel(writer_order, '%s' % cpn, index=True)

writer_user = pd.ExcelWriter('Cohort_Promo_by_User.xlsx')
writer_order = pd.ExcelWriter('Cohort_Promo_by_Order.xlsx')
writer_aos = pd.ExcelWriter('Cohort_Promo_by_AOS.xlsx')

for i in range(len(cpn)):
    # Orders where the coupon has been applied
    df = first_coupon[first_coupon['CouponCode']==cpn[i]]
    cust = df['lyty_card_no_y'].unique().tolist()
    # Retrieve all orders from these new customers
    ords = ecomm[ecomm['card_nbr'].isin(cust)]
    create_cohort(ords, cpn[i])
    
writer_user.save()
writer_order.save()
writer_aos.save()