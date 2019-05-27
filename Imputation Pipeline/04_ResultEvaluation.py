import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import date

os.chdir('C:/Users/lnwang/Documents/02Analysis/2. Analytical Project/'
         '201807 Simulation Method Enhancement/20181023_Indonesia')

featureSetName = 'missing_3M'
modelName = 'XGBoost'
folder_name = modelName + '_' + featureSetName

year = 2017
month = 6
periodicity = 'Q'
missingShopList = [999901]

round = 1

predict_level = ['PFC']
sheet_list = ['Total Market', 'Within Range Share', 'True0 PredN0', 'TrueN0 Pred0']

X1 = {}
X1['Period'] = []
X1['Total Market'] = []
for c in ['{}'.format(folder_name), 'GDG']:
    # X1['Total Market_{}'.format(c)] = []
    X1['Total Market Diff_{}'.format(c)] = []
    X1['MSE_{}'.format(c)] = []
    X1['Within Range Share_{}'.format(c)] = []
    X1['Within Range Share_PFC_{}'.format(c)] = []
    X1['True0 PredN0_{}'.format(c)] = []
    X1['TrueN0 Pred0_{}'.format(c)] = []

X2 = {}
X2['Period'] = []
X2['Scenario'] = []
X2['Total Market'] = []
X2['Total Market Diff'] = []
X2['MSE'] = []
X2['Within Range Share'] = []
X2['True0 PredN0'] = []
X2['TrueN0 Pred0'] = []

for r in range(round):

    if month>12:
        month = month-12
        year = year+1

    print('Reading prediction result......')
    X_test = pd.read_csv('./03 Feature/03-1 Test Feature/X_test_{}M{}.csv'.
                         format(year, month))

    y_test = pd.read_csv('./01 Data/01-1 Raw Input/realY.csv')
    y_test['date'] = pd.to_datetime(y_test['date'])
    y_test['date'] = y_test['date'].map(lambda x: x.date())
    tmp = date(year, month, 1)
    y_test = y_test[((y_test.phcode.isin(missingShopList)) &
                     (y_test.date == tmp))]
    y_test.rename(columns={'unit': 'y'}, inplace=True)

    y_pred = pd.read_csv('./04 Prediction/04-1 Prediction Result/{}/y_pred_{}M{}.csv'.
                         format(folder_name, year, month))
    y_pred.rename(columns={'y': 'y_{}'.format(folder_name)}, inplace=True)


    if month>=10:
        baseline = pd.read_table('./01 Data/01-2 GDG Result/ITMA_gdg_{}_M{}.txt'.format(year, month),
                                 delimiter=';')
    else:
        baseline = pd.read_table('./01 Data/01-2 GDG Result/ITMA_gdg_{}_M0{}.txt'.format(year, month),
                                 delimiter=';')
    baseline.rename(columns={'labcode': 'phcode',
                             'Units': 'y_GDG'}, inplace=True)
    baseline = baseline.groupby(['PFC'])[['y_GDG']].sum().reset_index()

    y = y_test[['phcode', 'PFC', 'y']]
    y = y.merge(y_pred, how='outer', on=['phcode', 'PFC']).fillna(0)
    y = y.merge(baseline[['PFC', 'y_GDG']], how='outer', on=['PFC']).fillna(0)
    y_PFC = y.groupby(['PFC'])[['y', 'y_{}'.format(folder_name), 'y_GDG']].sum()

    '''
        Regular absolute error measure
    '''
    # X1, for excel chart
    X1['Period'].append('{}M{}'.format(year, month))
    X1['Total Market'].append(sum(y['y']))
    for c in ['{}'.format(folder_name), 'GDG']:
        X1['Total Market Diff_{}'.format(c)].append(abs(sum(y['y_{}'.format(c)])/sum(y['y'])-1))
        X1['MSE_{}'.format(c)].append(mean_squared_error(y['y'], y['y_{}'.format(c)]))

    '''
        GDG error measure
    '''
    # a is predict value, b is true value
    def func1(a, b):
        if 0.8*b<=a<=1.2*b:
            return 1
        else:
            return 0

    def func2(a, b):
        if (a==0) & (b!=0):
            return 1
        elif (a!=0) & (b==0):
            return 0

    for c in ['{}'.format(folder_name), 'GDG']:
        y['within_20_{}'.format(c)] = y.apply(lambda x: func1(x['y_{}'.format(c)], x['y']), axis=1)
        y_PFC['within_20_PFC_{}'.format(c)] = y_PFC.apply(lambda x: func1(x['y_{}'.format(c)], x['y']), axis=1)
        y['true0_predN0_{}'.format(c)] = y.apply(lambda x: func2(x['y'], x['y_{}'.format(c)]), axis=1)
        y['trueN0_pred0_{}'.format(c)] = y.apply(lambda x: func2(x['y_{}'.format(c)], x['y']), axis=1)

        mask = y['within_20_{}'.format(c)] == 1
        X1['Within Range Share_{}'.format(c)].append(sum(y[mask]['y'])/sum(y['y']))
        mask = y_PFC['within_20_PFC_{}'.format(c)] == 1
        X1['Within Range Share_PFC_{}'.format(c)].append(sum(y_PFC[mask]['y'])/sum(y_PFC['y']))
        mask = y['true0_predN0_{}'.format(c)] == 1
        X1['True0 PredN0_{}'.format(c)].append(sum(y[mask]['y_{}'.format(c)])/sum(y['y_{}'.format(c)]))
        mask = y['trueN0_pred0_{}'.format(c)] == 1
        X1['TrueN0 Pred0_{}'.format(c)].append(sum(y[mask]['y']) / sum(y['y']))

    y = y.merge(X_test, how='left', on=predict_level)
    y.to_csv('./04 Prediction/04-1 Prediction Result/{}/y_result_{}M{}.csv'.
             format(folder_name, year, month), index=False)

    month = month + 1
    if month > 12:
        month = month - 12
        year = year + 1

'''
    Result Recording
'''
X1 = pd.DataFrame(X1)
filepath = './04 Prediction/04-1 Prediction Result/Result.xlsx'.format(folder_name)
isExist = os.path.exists(filepath)
if isExist:
    writer = pd.ExcelWriter(filepath)
    temp = pd.read_excel(writer, 'Total Market')
    temp = pd.merge(temp, X1[['Period', 'Total Market Diff_{}'.format(folder_name)]],
                    on='Period', how='outer')
    temp.to_excel(writer, 'Total Market', index=False)

    temp = pd.read_excel(writer, 'MSE')
    temp = pd.merge(temp, X1[['Period', 'MSE_{}'.format(folder_name)]], on='Period', how='outer')
    temp.to_excel(writer, 'MSE', index=False)

    temp = pd.read_excel(writer, 'Within Range')
    temp = pd.merge(temp, X1[['Period', 'Within Range Share_{}'.format(folder_name),
                              'Within Range Share_PFC_{}'.format(folder_name)]], on='Period', how='outer')
    temp.to_excel(writer, 'Within Range', index=False)

    temp = pd.read_excel(writer, 'True0 PredN0')
    temp = pd.merge(temp, X1[['Period', 'True0 PredN0_{}'.format(folder_name)]], on='Period', how='outer')
    temp.to_excel(writer, 'True0 PredN0', index=False)

    temp = pd.read_excel(writer, 'TrueN0 Pred0')
    X_col = [x for x in X1.columns if x not in temp.columns]
    temp = pd.merge(temp, X1[['Period', 'TrueN0 Pred0_{}'.format(folder_name)]], on='Period', how='outer')
    temp.to_excel(writer, 'TrueN0 Pred0', index=False)
    writer.close()
else:
    writer = pd.ExcelWriter(filepath)
    X1[['Period', 'Total Market', 'Total Market Diff_GDG', 'Total Market Diff_{}'.format(folder_name)]] \
        .to_excel(writer, 'Total Market', index=False)
    X1[['Period', 'MSE_GDG', 'MSE_{}'.format(folder_name)]]\
        .to_excel(writer, 'MSE', index=False)
    X1[['Period', 'Within Range Share_GDG', 'Within Range Share_{}'.format(folder_name),
        'Within Range Share_PFC_GDG', 'Within Range Share_PFC_{}'.format(folder_name)]] \
        .to_excel(writer, 'Within Range', index=False)
    X1[['Period', 'True0 PredN0_GDG', 'True0 PredN0_{}'.format(folder_name)]]\
        .to_excel(writer, 'True0 PredN0', index=False)
    X1[['Period', 'TrueN0 Pred0_GDG', 'TrueN0 Pred0_{}'.format(folder_name)]]\
        .to_excel(writer, 'TrueN0 Pred0', index=False)
    writer.close()