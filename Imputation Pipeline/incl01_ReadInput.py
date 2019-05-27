import pandas as pd
import datetime
import time
from datetime import date
import numpy as np


def combine_date(df, y, m):
    year = df[y]
    month = df[m]
    return date(int(year), int(month), 1)

def readInput(inputFile, priceFile, infoFile, clusterPFCfile, clusterBrickFile, clusterATCfile,
              readPrediction=False, folderName=None, year=None, month=None):

    train_ori = pd.read_csv(inputFile)
    # train_ori = pd.read_csv(inputFile, parse_dates=['date'])
    price = pd.read_csv(priceFile)
    info = pd.read_csv(infoFile)

    if readPrediction:
        y_pred = pd.read_csv('./04 Prediction/04-1 Prediction Result/{}/y_pred_{}M{}.csv'.
                             format(folderName, year, month))
        y_pred['date'] = date(year, month, 1)
        y_pred.rename(columns={'y': 'unit'}, inplace=True)
        train_ori = pd.concat([train_ori, y_pred], axis=0)

    # train_ori['date'] = train_ori.date.map(datetime.datetime.fromtimestamp)
    # train_ori['date'] = train_ori.date.map(datetime.date)
    train_ori['date'] = pd.to_datetime(train_ori['date'])
    train_ori['date'] = train_ori['date'].map(lambda x: x.date())

    price['date'] = price.apply(combine_date, axis=1, y='year', m='month')
    price.rename(columns={'pfc': 'PFC'}, inplace=True)
    info.rename(columns={'pfc': 'PFC'}, inplace=True)
    info['launch_year'] = info.launch_year. \
        apply(lambda x: 100 if x > 100 else x)

    print('Merging dataframes......')
    train_ori = train_ori.merge(info[['PFC', 'LabCode', 'atc4', 'launch_month', 'launch_year']],
                                on='PFC', how='left').fillna(0)
    if clusterPFCfile is not None:
        cluster_pfc = pd.read_table(clusterPFCfile, header=None, sep=',')
        cluster_pfc.rename(columns={0: 'PFC', 1: 'group_PFC'}, inplace=True)
        train_ori = train_ori.merge(cluster_pfc, on='PFC', how='left').fillna(9)
    if clusterBrickFile is not None:
        cluster_brick = pd.read_table(clusterBrickFile, header=None, sep=',')
        cluster_brick.rename(columns={0: 'brick', 1: 'group_brick'}, inplace=True)
        train_ori = train_ori.merge(cluster_brick, on='brick', how='left').fillna(9)
    if clusterATCfile is not None:
        cluster_atc = pd.read_table(clusterATCfile, header=None, sep=',')
        cluster_atc.rename(columns={0: 'atc4', 1: 'group_atc'}, inplace=True)
        train_ori = train_ori.merge(cluster_atc, on='atc4', how='left').fillna(99)

    return train_ori, price
