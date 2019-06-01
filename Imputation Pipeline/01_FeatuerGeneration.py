import os
import pandas as pd
from datetime import date
import numpy as np
from incl01_ReadInput import readInput
from incl02_IdentifyConstSupplier import idCnstShop
from incl03_Preprocess import preprocess
from incl04_FeatureCalculation import featureCalculation

###### ---Update below parameters--- ######

# work folder
os.chdir('C:/Users/lnwang/Documents/02Analysis/2. Analytical Project/'
         '201807 Simulation Method Enhancement/20181023_Indonesia')

# input, price, PFCinfo, cluster file path
inputFile = './01 Data/01-1 Raw Input/ind_input_missing.csv'
priceFile = './01 Data/01-1 Raw Input/ind_price_201504_201802.csv'
infoFile = './01 Data/01-1 Raw Input/pfc_info.csv'
# inputFile = './01 Data/01-1 Raw Input/lih_input_missing.csv'
# priceFile = './01 Data/01-1 Raw Input/lih_price_201501_201604.csv'
# infoFile = './01 Data/01-1 Raw Input/pfc_info.csv'
clustPFCfile = None
clustBrickfile = None
clustATCfile = None

featureSetName = 'missing'   # name to identify different feature set
resultName = 'missing_3M'
modelName = 'XGBoost'           # model name
folder_name = modelName + '_' + featureSetName
result_folder = modelName + '_' + resultName

type = 'missing'   # model train type - missing, notMissing, combined
isTrain = False   # True means extract
periodicity = 'Q'   # 'Q' for quarterly, 'M' for monthly
data_start_year = 2016   # feature start period - 3
data_start_month = 9
data_periods = 4

feature_start_year = 2017
feature_start_month = 6
feature_periods = 1
missingShopList = [999901]   # shop missing in feature start period
excludeShopList = [999901]

readPrediction = False   # whether need to read prediction result and set to input data,
                         # set it to True when prepare train data for model train type - missing & combined
readPred_year = 2017
readPred_month = 6

forecastLevel = ['phcode', 'PFC']

GROUPBY_CATEGORIES = [
    {'groupby': ['phcode', 'PFC'], 'name': 'phcode_PFC'},
    {'groupby': ['phcode'], 'name': 'phcode', 'number': 0},
    # {'groupby': ['brick'], 'name': 'brick', 'number': 1},
    {'groupby': ['atc4'], 'name': 'atc4', 'number': 2},
    {'groupby': ['LabCode'], 'name': 'LabCode', 'number': 3},
    {'groupby': ['PFC'], 'name': 'PFC', 'number': 4},
    # {'groupby': ['group_brick'], 'name': 'group_brick', 'number': 7},
    # {'groupby': ['group_atc'], 'name': 'group_atc', 'number': 8},
    # {'groupby': ['group_PFC'], 'name': 'group_PFC', 'number': 9},

    {'groupby': ['phcode', 'atc4'], 'name': 'phcode_atc'},
    {'groupby': ['phcode', 'LabCode'], 'name': 'phcode_labCode'},

    {'groupby': ['LabCode', 'atc4'], 'name': 'labCode_atc'}
]

growth_list = ['PFC', 'atc4', 'LabCode']

###### ---Finish Update--- ######

if isTrain:
    isExists = os.path.exists('./02 Model Training/02-1 Training Feature/{}'.format(featureSetName))
    if not isExists:
        os.makedirs('./02 Model Training/02-1 Training Feature/{}'.format(featureSetName))

train, price = readInput(inputFile, priceFile, infoFile, clustPFCfile, clustBrickfile, clustATCfile,
                         readPrediction, result_folder, readPred_year, readPred_month)

train = idCnstShop(train, data_start_year, data_start_month, data_periods, periodicity, missingShopList)

train = preprocess(train)

# # for debug
# train.to_csv("./test/train.csv", index=False)
# price.to_csv("./test/price.csv", index=False)
# train = pd.read_csv("./test/train.csv")
# train['date'] = pd.to_datetime(train['date'])
# train['date'] = train['date'].map(lambda x: x.date())
# price = pd.read_csv("./test/price.csv")


type_dict = {
    'missing': missingShopList,
    'combined': list(set([x for x in train['phcode']])),
    'notMissing': [x for x in list(set([x for x in train['phcode']]))
                   if x not in missingShopList and x != 10000]
}

featureCalculation(train, feature_periods, feature_start_year, feature_start_month, periodicity, isTrain,
                   3, [3], GROUPBY_CATEGORIES, growth_list,
                   type, missingShopList, excludeShopList, type_dict[type],
                   price, forecastLevel, featureSetName)

# def featureCalculation(train_ori, train_periods, train_start_year, train_start_month, periodicity, isTrain,
#                       constant_period, time_windows, GROUPBY_CATEGORIES, growth_list,
#                       type, missing_shop, exclude_shop, feature_shop,
#                       price, forecastLevel, featureSetName):
