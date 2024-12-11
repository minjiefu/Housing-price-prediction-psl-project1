# Step 0: Load necessary Python packages
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.linear_model import ElasticNet
np.random.seed(2809)

# MODEL XGB
# Step 1: Preprocess the training data

## load and preprocess the training data
df_train = pd.read_csv('train.csv')
X_train = df_train.drop(['PID', 'Sale_Price'], axis=1)
y_train = np.log(df_train['Sale_Price'])

## fill missing values in 'Garage_Yr_Blt' with 0
X_train['Garage_Yr_Blt']= X_train['Garage_Yr_Blt'].fillna(0)

## transform categorical variables to numerical variables
cat_columns = X_train.select_dtypes(include=['object']).columns
num_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
transformers=[
    ('num', 'passthrough', num_columns),  # Numerical columns remain unchanged
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)  # Categorical columns are one-hot encoded
])
X_train = preprocessor.fit_transform(X_train)

# Step 2: Fit Xgb  model
xgb_model = xgb.XGBRegressor(max_depth=6, eta=0.05, n_estimators=5000, subsample=0.6, random_state=2809)
xgb_model.fit(X_train, y_train)



# Step 3: Preprocess the test data
## Load and preprocess the test data
df_test = pd.read_csv('test.csv')
X_test = df_test.drop(['PID'], axis=1)
X_test['Garage_Yr_Blt']= X_test['Garage_Yr_Blt'].fillna(0)
X_test = preprocessor.transform(X_test)

# Step 4: Save predictions to files
## Make predictions using xgb model
xgb_pre = xgb_model.predict(X_test)


## Save predictions to file
pd.set_option('display.float_format', lambda x: '%.0f' % x)
output = pd.DataFrame(data={'PID': df_test['PID'], 'Sale_Price': np.exp(xgb_pre)})
output.to_csv("mysubmission1.txt",index=False, sep=',', header=True)


# MODEL Elastic Net
# Step 1: Preprocess the training data

## load and preprocess the training data
df_train2 = pd.read_csv('train.csv')
X_train2 = df_train2.drop(['PID', 'Sale_Price'], axis=1)
y_train2 = np.log(df_train2['Sale_Price'])

## fill missing values in 'Garage_Yr_Blt' with 0
X_train2['Garage_Yr_Blt']= X_train2['Garage_Yr_Blt'].fillna(0)
## remove
del_cols = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']
X_train2 = X_train2.drop(del_cols, axis=1)
## win
win_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
for var in win_cols:
    M = np.percentile(X_train2[var], 95)
    X_train2[var] = np.where(X_train2[var] > M, M, X_train2[var])
## transform categorical variables to numerical variables
cat_columns2 = X_train2.select_dtypes(include=['object']).columns
num_columns2 = X_train2.select_dtypes(include=['int64', 'float64']).columns
preprocessor2 = ColumnTransformer(
transformers=[
    ('num', 'passthrough', num_columns2),  # Numerical columns remain unchanged
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns2)  # Categorical columns are one-hot encoded
])
X_train2 = preprocessor2.fit_transform(X_train2)
## transform X_train2 back to dataframe
new_columns = list(num_columns2) + list(preprocessor2.named_transformers_['cat'].get_feature_names_out())
X_train2 = pd.DataFrame(X_train2.toarray(), columns=new_columns)

# Step 2: Fit elastic net model
elastic_net_model = ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=10000, random_state=2809)
elastic_net_model.fit(X_train2, y_train2)

# Step 3: Preprocess the test data
## Load and preprocess the test data
df_test2 = pd.read_csv('test.csv')
X_test2 = df_test2.drop(['PID'], axis=1)
X_test2['Garage_Yr_Blt']= X_test2['Garage_Yr_Blt'].fillna(0)
for var in win_cols:
    M = np.percentile(X_train2[var], 95)
    X_test2[var] = np.where(X_test2[var] > M, M, X_test2[var])

X_test2 = preprocessor2.transform(X_test2)
X_test2 = pd.DataFrame(X_test2.toarray(), columns=new_columns)
    


# Step 4: Save predictions to files
## Make predictions using xgb model
enet_pre = elastic_net_model.predict(X_test2)

## Save predictions to file
pd.set_option('display.float_format', lambda x: '%.0f' % x)
output = pd.DataFrame(data={'PID': df_test2['PID'], 'Sale_Price': np.exp(enet_pre)})
output.to_csv("mysubmission2.txt",index=False, sep=',', header=True)
