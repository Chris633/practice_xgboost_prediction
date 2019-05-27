import os
import pandas as pd
import xgboost as xgb

os.chdir('C:/Users/lnwang/Documents/02Analysis/2. Analytical Project/'
         '201807 Simulation Method Enhancement/20181023_Indonesia')

featureSetName = 'missing_3M'
modelName = 'XGBoost'
folder_name = modelName + '_' + featureSetName

train_periods = 3
train_start_year = 2016
train_start_month = 9
test_year = 2017
test_month = 6

forecast_level = ['phcode', 'PFC']
sheet_list = ['Total Market', 'Within Range Share', 'True0 PredN0', 'TrueN0 Pred0']

isExists = os.path.exists('./04 Prediction/04-1 Prediction Result/{}'.format(folder_name))
if not isExists:
    os.makedirs('./04 Prediction/04-1 Prediction Result/{}'.format(folder_name))

print('Reading train and test data......')
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

X_test = pd.read_csv('./03 Feature/03-1 Test Feature/X_test_{}M{}.csv'.
                      format(test_year, test_month))

print('Training Model.....')

predictor = [x for x in X_train.columns if x not in forecast_level]

regeressor = [
    # {'name': 'Lasso', 'func': linear_model.Lasso(alpha=30)},
    # {'name': 'XGBoost_general',
    #  'func': XGBRegressor(
    #     booster='gbtree',
    #     max_depth=5, # [1,...]
    #     min_child_weight=1,  # 叶子节点上所有样本权重和小于min_child_weight停止分裂,[0,...]
    #     gamma=0,   # 为了对树的叶子节点进一步分隔必须设置的损失减少的最小值，越小算法越保守，[0,...]
    #     subsample=0.8,  # 每棵树行随机采样的比例，过小会欠拟合，典型值0.5-1, (0,1]
    #     colsample_bytree=0.8,  # 每棵树列随机采样的比例，典型值0.5-1, (0,1]
    #     reg_lambda=1,  # L2正则化项
    #     reg_alpha=1,  # L1正则化项
    #     learning_rate=0.1,
    #     n_estimators=200,
    #     objective='reg:linear',
    #     seed=27)
    # },
    {'name': 'XGBoost_adjust',
     'func': xgb.Booster(model_file='./02 Model Training/02-3 Model/{}/XGBoost_{}M{}'.
                        format(folder_name, train_start_year, train_start_month))},

    # {'name': 'XGBoost_raw',
    #  'func': xgb.Booster(model_file='./02 Model Training/02-3 Model/{}/XGBoost_{}M{}_raw'.
    #                     format(folder_name, train_start_year, train_start_month))}

    # {'name': 'SVR', 'func': SVR(C=1.0, epsilon=0.2)}
    # {'name': 'NN', 'func': DeepMethod(X_train.values, y_train.values, X_val.values, 50)}
    # {'name': 'Ridge', 'func': (X_train.values, y_train.values, X_val.values, 50)}
]

for model in regeressor:
    if model['name']=='NN':
        y_val_pred = model['func']
    elif model['name']=='XGBoost_general':
        reg = model['func']
        reg.fit(X_train[predictor], y_train['y'], eval_metric='rmse')
        y_pred = reg.predict(X_test[predictor])
    elif model['name']=='XGBoost_adjust':
        featureSelected = pd.read_excel('./02 Model Training/02-2 Feature Importance/{}/Feature Importance.xlsx'.
                                      format(folder_name))
        featureSelected = list(set(featureSelected['index']))

        reg = model['func']
        data = xgb.DMatrix(X_test[featureSelected])
        y_pred = reg.predict(data)
    elif model['name']=='XGBoost_raw':
        reg = model['func']
        data = xgb.DMatrix(X_test[predictor])
        y_pred = reg.predict(data)


    y_pred = pd.DataFrame(y_pred)
    y_pred.iloc[:, 0] = y_pred.iloc[:, 0].map(lambda x: x if x>0 else 0)
    y = pd.DataFrame(X_test[forecast_level].values)
    y.columns = forecast_level
    y['y'] = y_pred

    y.to_csv('./04 Prediction/04-1 Prediction Result/{}/y_pred_{}M{}.csv'.
             format(folder_name, test_year, test_month), index=False)

