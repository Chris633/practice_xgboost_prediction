from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd

def constant_WHS_identification(df, start_date, periods, periodicity, missing):
    date_windows = []
    if periodicity=='M':
        date_windows = [start_date + relativedelta(months=i) for i in range(periods)]
    elif periodicity == 'Q':
        date_windows = [start_date + relativedelta(months=i*3) for i in range(periods)]
    print(date_windows)

    tmp = df[date_windows]
    tmp['min_sales'] = tmp.min(axis=1)
    mask = tmp['min_sales'] > 0
    cons = tmp[mask].reset_index()

    missingDf = pd.DataFrame(missing)
    if missing:
        missingDf.columns = ['phcode']
    phcode = pd.concat([cons[['phcode']], missingDf], axis=0)

    return phcode


def idCnstShop(train, start_year, start_month, periods, periodicity, missing):

    start_date = date(start_year, start_month, 1)
    end_date = start_date
    if periodicity=='Q':
        end_date = start_date + relativedelta(months=periods*3)
    elif periodicity=='M':
        end_date = start_date + relativedelta(months=periods)
    mask = (train['date'] <= end_date) & (train['date'] >= start_date)
    df = train.loc[mask]
    df = df[df['unit'] > 0]

    df = df.groupby(['phcode', 'date'])[['unit']].sum().unstack(level=-1).fillna(0)
    df.columns = df.columns.get_level_values(1)

    '''identify constant supplier'''
    constant = constant_WHS_identification(df, start_date, periods, periodicity, missing)

    train = train.merge(constant, on='phcode', how='right')

    def changeShop(x, missing):
        if x not in missing and x not in [999901, 999902, 999903, 999904]:
            return 10000
        else:
            return x
    train['phcode'] = train.apply(lambda x: changeShop(x.phcode, missing), axis=1)

    column = [x for x in train.columns if x!='unit']
    train = train.groupby(column)[['unit']].sum()
    train = train.reset_index()

    return train




