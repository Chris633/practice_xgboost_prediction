import os
import pandas as pd
from incl05_XGBoost_Auto import XGBoost_ParamOpti

os.chdir('C:/Users/lnwang/Documents/02Analysis/2. Analytical Project/'
         '201807 Simulation Method Enhancement/20181023_Indonesia')

type_list = ['missing',
             'non-missing',
             'combined']
# The feature set name contains all related to feature, including periods, missing/non-missing/combined,
#   feature selection
featureSetName = 'missing_3M'
folder_name = 'XGBoost_' + featureSetName
train_year = 2016
train_month = 9
predict_level = ['phcode', 'FCC']  # update if necessary

X_train_files = [
    './02 Model Training/02-1 Training Feature/missing/X_train_2016M9.csv',
    './02 Model Training/02-1 Training Feature/missing/X_train_2016M12.csv',
    './02 Model Training/02-1 Training Feature/missing/X_train_2017M3.csv'
]
X_train_list = []
for f in X_train_files:
    X_train_list.append(pd.read_csv(f))
X_train = pd.concat(X_train_list, axis=0)

y_train_files = [
    './02 Model Training/02-1 Training Feature/missing/y_train_2016M9.csv',
    './02 Model Training/02-1 Training Feature/missing/y_train_2016M12.csv',
    './02 Model Training/02-1 Training Feature/missing/y_train_2017M3.csv'
]
y_train_list = []
for f in y_train_files:
    y_train_list.append(pd.read_csv(f))
y_train = pd.concat(y_train_list, axis=0)

# Create necessary folders
isExists = os.path.exists('./02 Model Training/02-2 Feature Importance/{}'.format(folder_name))
if not isExists:
    os.makedirs('./02 Model Training/02-2 Feature Importance/{}'.format(folder_name))
isExists = os.path.exists('./02 Model Training/02-3 Model/{}'.format(folder_name))
if not isExists:
    os.makedirs('./02 Model Training/02-3 Model/{}'.format(folder_name))

# Feature importance calculation using raw XGBoost model
feat_imp = XGBoost_ParamOpti(X_train, y_train, train_year, train_month,
                             folder_name, predict_level,
                             False)

# Keep features with score above average
averageScore = feat_imp['Importance Score'].mean()
feat_imp['flag'] = feat_imp['Importance Score'].apply(lambda x: 1 if x > averageScore else 0)
feat_imp.sort_values('Importance Score', inplace=True, ascending=False)
feat_imp = feat_imp[feat_imp['flag']==1]

aboveAverage = list(set(feat_imp['index']))

XGBoost_ParamOpti(X_train[aboveAverage], y_train, train_year, train_month,
                  folder_name, predict_level,
                  True)

# def XGBoost_ParamOpti(X_train, y_train, train_year, train_month,
#                       folder_name,
#                       gridSearch=True):