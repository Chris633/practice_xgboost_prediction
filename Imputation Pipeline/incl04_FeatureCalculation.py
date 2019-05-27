import time
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd

'''
    train:             input file to generate training set
    train_periods:     combine n feature periods in 1 training set
    train_start_year:  year of first period in training set
    train_start_month: month of first period in training set
    isTrain:           True means to create y_train; False means only generate feature set
    
    constant_period:   rule to identify constant shop
    time_windows:      calculate general feature based on back n periods
    
    forecastLevel:     forecast level, eg. ['phcode', 'brick', 'PFC'] same as in GROUPBY_CATEGORIES
    
    exclude_shop:      exclude shops in this list from growth calculation
                       be sure to always include missing shop in this list!!!
    feature_shop:      calculate feature for which shops, missing, non-mnissing or all
    
    price:             the price dataset
    featureSetName:    name of this feature set, related to scenario
    
    还需要对missing shop, feature shop做调整，保留missing或者non missing或者all feature
'''

def featureCalculation(train_ori, train_periods, train_start_year, train_start_month, periodicity, isTrain,
                      constant_period, time_windows, GROUPBY_CATEGORIES, growth_list,
                      type, missing_shop, exclude_shop, feature_shop,
                      price, forecastLevel, featureSetName):

    def get_timespan(df, dt, minus, periods, periodicity):
        date_windows=[]
        if periodicity=='M':
            date_windows = [dt - relativedelta(months=minus-i) for i in range(periods)]
        elif periodicity == 'Q':
            date_windows = [dt - relativedelta(months=(minus-i)*3) for i in range(periods)]
        print(date_windows)
        return df[date_windows]


    # return list of WHS code which is constant and not missing
    def constant_WHS_identification(df, cur, exclude, periodicity):
        tmp = get_timespan(df, cur, constant_period, constant_period + 1, periodicity)
        tmp['min'] = tmp.min(axis=1)
        mask = tmp['min'] > 0
        cons = tmp[mask]
        cons = cons.index.tolist()
        return [x for x in cons if x not in exclude]


    def constant_growth(df, cons, level, cur, name_prefix, periodicity):
        tmp = df[df.phcode.isin(cons)].groupby(level + ['date'])[['unit']].sum().unstack(
            level=-1).fillna(0)
        tmp.columns = tmp.columns.get_level_values(1)
        tmp = get_timespan(tmp, cur, constant_period, constant_period + 1, periodicity)

        X = {}
        X['back_avg'] = tmp.iloc[:, :constant_period].mean(axis=1).values
        X['minus'] = (tmp.iloc[:, constant_period] - X['back_avg']).values
        X['back_avg'][X['back_avg'] == 0] = 0.1
        X['growth'] = X['minus'] / X['back_avg']

        X = pd.DataFrame(X)
        X.columns = ['{}_{}'.format(name_prefix, c) for c in X.columns]
        X.index = tmp.index

        if name_prefix == 'brick':
            quantile_10 = X['brick_growth'].quantile(0.1, interpolation='higher')
            quantile_90 = X['brick_growth'].quantile(0.9, interpolation='lower')
            X['brick_growth'] = X['brick_growth'].map(lambda x: quantile_10 if x < quantile_10 else x)
            X['brick_growth'] = X['brick_growth'].map(lambda x: quantile_90 if x > quantile_90 else x)
        return X['{}_growth'.format(name_prefix)]


    def history_feature(df, date, name_prefix, periodicity, is_train=False):
        X = {}
        for i in time_windows:
            tmp = get_timespan(df, date, i, i, periodicity)
            X['diff_{}_mean'.format(i)] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_{}'.format(i)] = tmp.mean(axis=1).values
            X['median_{}'.format(i)] = tmp.median(axis=1).values
            X['min_{}'.format(i)] = tmp.min(axis=1).values
            X['max_{}'.format(i)] = tmp.max(axis=1).values
            X['std_{}'.format(i)] = tmp.std(axis=1).values
            X['has_sales_month_in_last_{}'.format(i)] = (tmp > 0).sum(axis=1).values
        for i in range(1, max(time_windows) + 1):
            X['sales_M{}'.format(i)] = get_timespan(df, date, i, 1, periodicity).values.ravel()

        X = pd.DataFrame(X)

        X.columns = ['{}_{}'.format(name_prefix, c) for c in X.columns]

        if is_train:
            t = pd.date_range(date, periods=1)
            y = df[t].values
            return X, y
        else:
            return X


    # def share_feature(df, level1, level2, periodicity, missing):
    #     tmp = get_timespan(df, cur, 3, 3, periodicity)
    #     df = df1_missing.merge()


    def prepare_dataset(cur, m_delta, train_periods, supplier,
                        df_dict, train, GROUPBY_CATEGORIES, growth_list,
                        is_train=False):

        # create feature on different groupby levels
        for spec in GROUPBY_CATEGORIES:
            if spec['name'] == 'phcode':
                '''Since there is no shop level feature for missing supplier, 
                   no shop level feature is available'''
                pass
            else:
                print('{} level feature generation......{} of {}'.format(spec['name'], m_delta + 1, train_periods))
                '''
                    The rebuilt feature dataset has no index. 
                    It is needed to assign it to correct index level - df_dict[spec['name']]
                    and map it to forecast level - train.
    
                    X_tmp1 : history feature
                    X_tmp2 : back 3 average * PFC/brick level growth
                    X_tmp  : concat of X_tmp1 and X_tmp2
                    X_k    : concat of different groupby levels
                '''
                # History feature
                if spec['groupby'] == forecastLevel and is_train:
                    X_tmp1, y_tmp = history_feature(df_dict[spec['name']], cur,
                                                    spec['name'], periodicity, True)

                    y_tmp = pd.DataFrame(y_tmp)
                    y_tmp.columns = ['y']
                    y_tmp.index = df_dict[spec['name']].index
                    y_tmp_index = train.reset_index()[spec['groupby']]
                    y_tmp = y_tmp.reindex(y_tmp_index)
                    y_tmp.reset_index(inplace=True)

                else:
                    X_tmp1 = history_feature(df_dict[spec['name']], cur,
                                             spec['name'], periodicity, False)

                X_tmp1.index = df_dict[spec['name']].index

                if len(spec['groupby']) == 1:
                    names = train.index.names
                    for n in range(len(names)):
                        if spec['name']==names[n]:
                            number = n
                    X_tmp1_index = train.index.get_level_values(number)
                else:
                    X_tmp1_index = train.reset_index()[spec['groupby']]

                X_tmp1 = X_tmp1.reindex(X_tmp1_index)
                X_tmp = X_tmp1.reset_index(drop=True)

                # Growth feature
                # calculate back 3 average * PFC/brick level growth
                if spec['name'] in growth_list:
                    tmp = train.stack().reset_index().rename(columns={0: 'unit'})
                    X_tmp2 = constant_growth(tmp, constant, spec['groupby'], cur, spec['name'], periodicity)

                    if len(spec['groupby']) == 1:
                        names = train.index.names
                        for n in range(len(names)):
                            if spec['name'] == names[n]:
                                number = n
                        X_tmp2_index = train.index.get_level_values(number)
                    else:
                        X_tmp2_index = train.reset_index()[spec['groupby']]

                    X_tmp2 = X_tmp2.reindex(X_tmp2_index)
                    X_tmp2 = X_tmp2.reset_index(drop=True)

                    X_tmp = pd.concat([X_tmp, X_tmp2], axis=1)

                # concat different level: X_k
                if spec == GROUPBY_CATEGORIES[0]:
                    X_k = X_tmp
                else:
                    X_k = pd.concat([X_k, X_tmp], axis=1)

        X_k.index = train.index
        X_k = X_k.reset_index()

        forecastLevelName = forecastLevel[0]
        for x in range(1, len(forecastLevel)):
            forecastLevelName = forecastLevelName + '_' + forecastLevel[x]
        for name in growth_list:
            X_k['{}_growth_est'.format(name)] = X_k['{}_mean_{}'.format(forecastLevelName, constant_period)] \
                                                * (1 + X_k['{}_growth'.format(name)]).fillna(0)
            del X_k['{}_growth'.format(name)]

        # Only keep features for target supplier
        X_k = X_k[X_k.phcode.isin(supplier)]

        if is_train:
            y_tmp = y_tmp[y_tmp.phcode.isin(supplier)]

        if is_train:
            return X_k, y_tmp
        else:
            return X_k

    print('Preparing training dataset.....')

    base_periods = max(train_periods, constant_period)
    date_base = date(train_start_year, train_start_month, 1)
    X_l = []
    y_l = []
    for m_delta in range(train_periods):
        cur = date_base + relativedelta(months=m_delta)
        print(cur)

        if (type=='combined') and (m_delta==train_periods-1) and isTrain:
            feature_shop = [x for x in feature_shop if x not in missing_shop]

        if periodicity=='Q':
            tmp1 = date(train_start_year, train_start_month, 1) + relativedelta(months=m_delta*3)
            tmp2 = tmp1 - relativedelta(months=base_periods*3)
        elif periodicity=='M':
            tmp1 = date(train_start_year, train_start_month, 1) + relativedelta(months=m_delta)
            tmp2 = tmp1 - relativedelta(months=base_periods)
        mask = (train_ori['date'] <= tmp1) & (train_ori['date'] >= tmp2)
        train = train_ori.loc[mask]
        train = train[train['unit'] > 0]

        mask = [x for x in train.columns if x not in ['date', 'unit']]
        train = train.groupby(mask + ['date'])[['unit']].sum()

        df_dict = {}
        for spec in GROUPBY_CATEGORIES:
            print('{} level groupby......'.format(spec['name']))
            df = train.groupby(spec['groupby'] + ['date'])[['unit']].sum().unstack(
                level=-1).fillna(0)
            df.columns = df.columns.get_level_values(1)
            df_dict[spec['name']] = df

        train = train.unstack().fillna(0)
        train.columns = train.columns.get_level_values(1)

        '''identify constant WHS'''
        constant = constant_WHS_identification(df_dict['phcode'], cur, exclude_shop, periodicity)

        '''feature generation'''
        if isTrain:
            X_k, y_tmp = prepare_dataset(cur, m_delta, train_periods, feature_shop,
                                         df_dict, train, GROUPBY_CATEGORIES, growth_list, isTrain)
            y_l.append(y_tmp)
            print(y_tmp.shape)
        else:
            X_k = prepare_dataset(cur, m_delta, train_periods, feature_shop,
                                         df_dict, train, GROUPBY_CATEGORIES, growth_list, isTrain)
        tmp = date(cur.year, cur.month, 1)
        X_k = X_k.merge(price.loc[price['date'] == tmp][['PFC', 'price']], on='PFC', how='left').fillna(0)

        '''X_l : list to collect different training periods features'''
        X_l.append(X_k)
        print(X_k.shape)

    '''X_train : concat of different training periods features'''
    X_train = pd.concat(X_l, axis=0)
    X_train['PFC_growth_est'].fillna(0, inplace=True)

    if isTrain:
        X_train.reset_index(drop=True).to_csv('./02 Model Training/02-1 Training Feature/{}/X_train_{}M{}.csv'.
                                              format(featureSetName, train_start_year,
                                                     train_start_month), index=False)
    else:
        X_train.reset_index(drop=True).to_csv('./03 Feature/03-1 Test Feature/X_test_{}M{}.csv'.
                                              format(train_start_year, train_start_month), index=False)

    if isTrain:
        y_train = pd.concat(y_l, axis=0)
        y_train.reset_index(drop=True).to_csv('./02 Model Training/02-1 Training Feature/{}/y_train_{}M{}.csv'.
                                              format(featureSetName, train_start_year,
                                                     train_start_month), index=False)



