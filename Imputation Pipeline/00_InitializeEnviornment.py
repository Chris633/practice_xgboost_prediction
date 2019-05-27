import os
import pandas as pd
from datetime import date
from datetime import date
from dateutil.relativedelta import relativedelta

###### ---Update below parameters--- ######

# Work folder, sub folders will be created under this folder.
os.chdir('C:/Users/lnwang/Documents/02Analysis/2. Analytical Project/'
         '201807 Simulation Method Enhancement/20181023_Indonesia')

# missing dictionary:
#   {missing shop code : [start missing year, start missing month]}
# this is used to delete data from missing shop and create missingInput
missing = {
    999901: [2017, 6],
    999902: [2017, 9],
    999903: [2017, 12],
    999904: [2018, 3]
}
# missing = {
#     1017: [2015, 7]
# }

# input file to be loaded
rawInput = './01 Data/01-1 Raw Input/ind_input_201504_201802.sas7bdat'
# rawInput = './01 Data/01-1 Raw Input/lih_input_201501_201604.sas7bdat'

# file name of missing input file, this is to be created in this code
missingInput = './01 Data/01-1 Raw Input/ind_input_missing.csv'
# missingInput = './01 Data/01-1 Raw Input/lih_input_missing.csv'

realInput = './01 Data/01-1 Raw Input/realY.csv'

###### ---Finish Update--- ######

folder = {'01 Data': ['Raw Input', 'GDG Result', 'Cluster'],
          '02 Model Training': ['Training Feature', 'Feature Importance', 'Model'],
          '03 Feature': ['Test Feature'],
          '04 Prediction': ['Prediction Result', 'Comparison'],
          '05 Other': ['Version']}

def CreatingFolder(folder):
    for f1 in folder:
        for f2 in range(len(folder[f1])):
            isExists = os.path.exists('./{}/{}-{} {}'.format(f1, f1[:2], f2+1, folder[f1][f2]))
            if not isExists:
                os.makedirs('./{}/{}-{} {}'.format(f1, f1[:2], f2+1, folder[f1][f2]))
                print('Creating folder ./{}/{}-{} {}'.format(f1, f1[:2], f2+1, folder[f1][f2]))


def InitializeRawData(miss_dict, inputFile, outputFile):
    raw_input = pd.read_sas(inputFile)
    # raw_input['brick'] = raw_input['brick'].map(lambda x: int(x.decode())).astype(np.uint16)
    raw_input.rename(columns={'pfc': 'PFC'}, inplace=True)
    # raw_input['PFC'] = raw_input['PFC'].map(lambda x: int(x.decode())).astype(np.uint16)
    raw_input['PFC'] = raw_input['PFC'].map(lambda x: int(x))

    def combine_date(df, y, m):
        year = df[y]
        month = df[m]
        return date(int(year), int(month), 1)

    raw_input['date'] = raw_input.apply(combine_date, axis=1, y='year', m='month')
    input_copy = raw_input
    realY_list = []
    for s in miss_dict:
        missing_date = date(miss_dict[s][0], miss_dict[s][1], 1)
        input_copy = input_copy[~((input_copy.phcode==s) & (input_copy.date >= missing_date))]
        realY_list.append(raw_input[((raw_input.phcode==s) & (raw_input.date >= missing_date))])

    input_copy.drop(['year', 'month'], axis=1, inplace=True)
    input_copy.to_csv(outputFile, index=False)
    realY = pd.concat(realY_list, axis=0)
    realY.drop(['year', 'month'], axis=1, inplace=True).to_csv(realInput, index=False)

# step 1, create folders
# CreatingFolder(folder)

# step 2, create input data without missing suppliers
InitializeRawData(missing, rawInput, missingInput)