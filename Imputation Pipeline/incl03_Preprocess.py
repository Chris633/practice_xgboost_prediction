from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(train):
    print('Category feature encoding......')
    le = LabelEncoder()
    train['atc4'] = le.fit_transform(train['atc4'].values)
    train['LabCode'] = train['LabCode'].astype(np.uint16)

    return train