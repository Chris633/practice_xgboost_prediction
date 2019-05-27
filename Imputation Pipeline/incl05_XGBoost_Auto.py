import os
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import copy
import time

def XGBoost_ParamOpti(X_train, y_train, train_year, train_month,
                      folder_name, exclude_level,
                      gridSearch=True):
    param0 = [                # parameter testing range
        # 0
        {'learning_rate': 0.1,
         'n_estimators': 10000},
        # 1
        {'max_depth': range(1, 12, 2),
         'min_child_weight': range(1, 12, 2)},
        # 2
        {'gamma': [i / 10.0 for i in range(0, 5)]},
        # 3
        {'subsample': [i / 10.0 for i in range(6, 10)],
         'colsample_bytree': [i / 10.0 for i in range(6, 10)]},
        # 4
        {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]},
        # 5
        {'reg_lambda': [1e-5, 1e-2, 0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]}
    ]

    '''
        modelfitCV: Used to adjust n_estimator and show rmse change in train & test
        input:
            alg, model (XGBRegressor)
            dtrain, train set (DataFrame)
            dtest, test set (DataFrame)
            predictors, feature for model (List)
            target, y (prediction target) (List)
            cv_folds, k-folds cross validation
            early_stopping_rounds, default 20
        output:
            cvresult.shape[0], n_estimator for input model
            mse, MSE for input dtest
    '''
    def modelfitCV(alg, dtrain, dtest, predictors, target, cv_folds=5, early_stopping_rounds=20):
        # Cross validation to get best n_estimators
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False,
                          verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(dtest[predictors], dtest[target], eval_metric='rmse')

        # Predict training set:
        dtest_predictions = alg.predict(dtest[predictors])

        # Print model report:
        mse = metrics.mean_squared_error(dtest[target].values, dtest_predictions)
        print("\nModel Report")
        print("Explained Variance Score (Test) : %.4g" %
              metrics.explained_variance_score(dtest[target].values, dtest_predictions))
        print("MSE (Test): %f" % mse)

        # Show feature importance
        # feat_imp.plot(kind='bar', title='Feature Importances')
        # plt.ylabel('Feature Importance Score')
        # plt.show()
        return cvresult.shape[0], mse

    '''
        gridSearchParam: gridSearch, return best_params_
    '''
    def gridSearchParam(model, param, X, y):
        gsearch = GridSearchCV(estimator=model, param_grid=param,
                               scoring='neg_mean_squared_error', n_jobs=1,
                               iid=False, cv=5)
        gsearch.fit(X, y)
        return gsearch.best_params_


    '''
        exportResult: export result to excel
    '''
    def exportResult(filePath, contentFrame):
        isExist = os.path.exists(filePath)
        # writer = pd.ExcelWriter(filePath)
        if isExist:
            oldFrame = pd.read_excel(filePath)
            newFrame = pd.concat([oldFrame, contentFrame])
            newFrame.to_excel(filePath, index=False)
        else:
            contentFrame.to_excel(filePath, index=False)
        # writer.close()


    '''
        cleanList: when adjusting parameter in smaller step, 
                   this function guarantees parameter is in legal range
        option: 1 for parameter [0, ...)
                2 for parameter [1, ...)
                3 for parameter (0, 1]
    '''
    def cleanList(paramList, option):
        new_list = []
        if option == 1:
            new_list = [x for x in paramList if x >= 0]
        if option == 2:
            new_list = [x for x in paramList if x >= 1]
        if option == 3:
            new_list = [x for x in paramList if ((x <= 1)and(x > 0))]
        return new_list


    if gridSearch is False:

        print('Period:    {}M{}'.format(train_year, train_month))

        xgb1 = XGBRegressor(
            booster='gbtree',
            max_depth=5,  # [1,...]
            min_child_weight=1,  # 叶子节点上所有样本权重和小于min_child_weight停止分裂,[0,...]
            gamma=0,  # 为了对树的叶子节点进一步分隔必须设置的损失减少的最小值，越小算法越保守，[0,...]
            subsample=0.8,  # 每棵树行随机采样的比例，过小会欠拟合，典型值0.5-1, (0,1]
            colsample_bytree=0.8,  # 每棵树列随机采样的比例，典型值0.5-1, (0,1]
            reg_lambda=1,  # L2正则化项
            reg_alpha=1,  # L1正则化项
            learning_rate=0.1,
            n_estimators=10000,
            objective='reg:linear',
            seed=27
        )

        predictor = [x for x in X_train.columns if x not in exclude_level]
        target = ['y']
        dtrain = pd.concat([X_train, y_train[target]], axis=1)
        n_estimator, mse = modelfitCV(xgb1, dtrain, dtrain, predictor, target)
        xgb1.set_params(n_estimators=n_estimator)
        xgb1.save_model('./02 Model Training/02-3 Model/{}/XGBoost_{}M{}_raw'.
                        format(folder_name, train_year, train_month))

        feat_imp = pd.Series(xgb1.get_booster().get_fscore())
        feat_imp = pd.DataFrame(feat_imp)
        feat_imp.rename(columns={0: 'Importance Score'}, inplace=True)
        feat_imp.sort_values('Importance Score', ascending=False, inplace=True)
        feat_imp['Period'] = '{}M{}'.format(train_year, train_month)
        feat_imp['Model'] = 'XGBoost'
        feat_imp.reset_index(inplace=True)
        exportResult('./02 Model Training/02-2 Feature Importance/{}/Feature Importance Raw.xlsx'.
                     format(folder_name), feat_imp)
        return feat_imp

    else:

        print('Period:    {}M{}'.format(train_year, train_month))

        #  A general parameter setting for XGBoost model.
        #  gridSearch will update it to optimized param and save corresponding model.
        xgb1 = XGBRegressor(
            booster='gbtree',
            max_depth=5, # [1,...]
            min_child_weight=1,  # 叶子节点上所有样本权重和小于min_child_weight停止分裂,[0,...]
            gamma=0,   # 为了对树的叶子节点进一步分隔必须设置的损失减少的最小值，越小算法越保守，[0,...]
            subsample=0.8,  # 每棵树行随机采样的比例，过小会欠拟合，典型值0.5-1, (0,1]
            colsample_bytree=0.8,  # 每棵树列随机采样的比例，典型值0.5-1, (0,1]
            reg_lambda=1,  # L2正则化项
            reg_alpha=1,  # L1正则化项
            learning_rate=0.1,
            n_estimators=10000,
            objective='reg:linear',
            seed=27
        )
        param1 = copy.deepcopy(param0)

        predictor = [x for x in X_train.columns if x not in exclude_level]

        target = ['y']
        dtrain = pd.concat([X_train, y_train[target]], axis=1)

        '''
            GridSearch:
                step 0: set common value to each parameter; use gridSearch=False to get n_estimators for learning rate
                step 1: use gridSearch=True to get max_depth & min_child_weight
                step 2: use gridSearch=True to get gama
                step 3: use gridSearch=False to re-adjust n_estimators for new set of parameters
                step 4: use gridSearch=True to get colsample_bytree and subsample
                step 5: use gridSearch=True to get reg_alpha and reg_lambda
                step 6: adjust learning rate, then re-adjust n_estimators
            分成种操作：
                单/多个参数调整，参数名，调整范围
                是否需要进一步微调，微调的范围
        '''

        '''
            Step 0
            set common value to each parameter; 
            use gridSearch=False to get n_estimators for learning rate
        '''
        n_estimator1, mse1 = modelfitCV(xgb1, dtrain, dtrain, predictor, target)
        xgb1.set_params(n_estimators=n_estimator1)
        '''
            Step 1
            use gridSearch=True to get max_depth & min_child_weight
        '''
        print(' Step 1.1: Optimize max_depth & min_child_weight by gap 2...')
        start_time = time.time()
        best_param = gridSearchParam(xgb1, param1[1], dtrain[predictor], dtrain[target])
        param1[1]['max_depth'] = cleanList([best_param['max_depth'] - 1,
                                            best_param['max_depth'],
                                            best_param['max_depth'] + 1], 2)
        param1[1]['min_child_weight'] = cleanList([best_param['min_child_weight'] - 1,
                                                   best_param['min_child_weight'],
                                                   best_param['min_child_weight'] + 1], 1)
        print("     Best parameter is ", best_param)
        end_time = time.time()
        time_taken = end_time - start_time
        print(' Step 1.1 took', time_taken, 'seconds')

        print(' Step 1.2: Optimize max_depth & min_child_weight by gap 1...')
        best_param = gridSearchParam(xgb1, param1[1], dtrain[predictor], dtrain[target])
        param1[1]['max_depth'] = best_param['max_depth']
        param1[1]['min_child_weight'] = best_param['min_child_weight']
        print("     Best parameter is ", best_param)
        xgb1.set_params(max_depth=best_param['max_depth'],
                        min_child_weight=best_param['min_child_weight'])
        '''
            Step 2
            use gridSearch=True to get gamma
        '''
        print(' Step 2: Optimize gamma...')
        best_param = gridSearchParam(xgb1, param1[2], dtrain[predictor], dtrain[target])
        param1[2]['gamma'] = cleanList([i / 100.0 for i in
                                        range(int(best_param['gamma'] * 100 - 5),
                                              int(best_param['gamma'] * 100 + 6), 5)], 1)
        print("     Best parameter is ", best_param)

        best_param = gridSearchParam(xgb1, param1[2], dtrain[predictor], dtrain[target])
        param1[2]['gamma'] = best_param['gamma']
        print("     Best parameter is ", best_param)
        xgb1.set_params(gamma=best_param['gamma'])
        '''
            Step 3
            use gridSearch=False to re-adjust n_estimators for new set of parameters
        '''
        print(' Step 3: Adjust n_estimators...')
        xgb1.set_params(n_estimators=10000)
        n_estimator2, mse2 = modelfitCV(xgb1, dtrain, dtrain, predictor, target)
        xgb1.set_params(n_estimators=n_estimator2)
        '''
            Step 4
            use gridSearch=True to get colsample_bytree and subsample
        '''
        # Adjust parameter by 1
        print(' Step 4.1: Optimize colsample_bytree and subsample by gap 1...')
        best_param = gridSearchParam(xgb1, param1[3], dtrain[predictor], dtrain[target])
        param1[3]['subsample'] = cleanList([i / 100.0 for i in
                                            range(int(best_param['subsample'] * 100) - 5,
                                                  int(best_param['subsample'] * 100) + 6, 5)], 3)
        param1[3]['colsample_bytree'] = cleanList([i / 100.0 for i in
                                                   range(int(best_param['colsample_bytree'] * 100) - 5,
                                                         int(best_param['colsample_bytree'] * 100) + 6, 5)], 3)
        print("     Best parameter is ", best_param)

        # Adjust parameter by 0.05
        print(' Step 4.2: Optimize colsample_bytree and subsample by gap 0.05...')
        best_param = gridSearchParam(xgb1, param1[3], dtrain[predictor], dtrain[target])
        param1[3]['subsample'] = best_param['subsample']
        param1[3]['colsample_bytree'] = best_param['colsample_bytree']
        print("     Best parameter is ", best_param)
        xgb1.set_params(subsample=best_param['subsample'],
                        colsample_bytree=best_param['colsample_bytree'])
        '''
            Step 5
            use gridSearch=True to get reg_alpha and reg_lambda
        '''
        print(' Step 5.1: Optimize reg_alpha...')
        best_param = gridSearchParam(xgb1, param1[4], dtrain[predictor], dtrain[target])
        print("     Best parameter is ", best_param)
        xgb1.set_params(reg_alpha=best_param['reg_alpha'])

        print(' Step 5.2: Optimize reg_lambda...')
        best_param = gridSearchParam(xgb1, param1[5], dtrain[predictor], dtrain[target])
        print("     Best parameter is ", best_param)
        xgb1.set_params(reg_lambda=best_param['reg_lambda'])
        '''
            Step 6
            adjust learning rate, then re-adjust n_estimators
        '''
        print(' Step 6: Change learning rate to 0.01 and adjust n_estimators...')
        xgb1.set_params(learning_rate=0.01, n_estimators=10000)
        print("Final model is ", xgb1)
        n_estimator3, mse3 = modelfitCV(xgb1, dtrain, dtrain, predictor, target)
        xgb1.set_params(n_estimators=n_estimator3)

        # Save model and export optimized parameter
        xgb1.save_model('./02 Model Training/02-3 Model/{}/XGBoost_{}M{}'.
                        format(folder_name, train_year, train_month))

        tmp = xgb1.get_xgb_params()
        tmp['Period'] = '{}M{}'.format(train_year, train_month)
        tmp['n_estimater2'] = n_estimator2
        tmp['n_estimater3'] = n_estimator3
        tmp['mse1'] = mse1
        tmp['mse2'] = mse2
        tmp['mse3'] = mse3
        tmp = pd.DataFrame(tmp, index=[0])
        print(' Optimized parameter for period {}M{} are:\n {}'.
              format(train_year, train_month, tmp))
        exportResult('./02 Model Training/02-3 Model/{}/Optimized Parameter.xlsx'.format(folder_name),
                     tmp)

        # Export feature importance
        feat_imp = pd.Series(xgb1.get_booster().get_fscore())
        feat_imp = pd.DataFrame(feat_imp)
        feat_imp.rename(columns={0: 'Importance Score'}, inplace=True)
        feat_imp.sort_values('Importance Score', ascending=False, inplace=True)
        feat_imp['Period'] = '{}M{}'.format(train_year, train_month)
        feat_imp['Model'] = 'XGBoost'
        feat_imp.reset_index(inplace=True)
        exportResult('./02 Model Training/02-2 Feature Importance/{}/Feature Importance.xlsx'.
                     format(folder_name), feat_imp)
