import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import seaborn as sns
import tsfresh.feature_extraction.feature_calculators as feature_calculator
from scipy.stats import normaltest

import warnings
warnings.filterwarnings('ignore')


@contextmanager
def timer(name):
    t = time.time()
    yield
    print('[%s] done in %.2f seconds' % (name, time.time() - t))
    

def make_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def agg_first_value(x):
    return x.values[0]


def agg_kurtosis(x):
    return feature_calculator.kurtosis(x)


def agg_normaltest_pvalue(x):
    if len(x) < 8:
        return np.nan
    return normaltest(x)[1]

def agg_num_bins(x):
    bins, _ = np.histogram(x, bins=10)
    return np.sum(bins>0)

def apply_corr(group):
    x = group['发动机转速']
    y = group['油泵转速']
    corr = pearsonr(x,y)
    return corr[0]


def make_features(df):
    feat_df = pd.DataFrame(df.sample_file_name.unique(),
                           columns=['sample_file_name'])
    
    fea = df.groupby('sample_file_name')['活塞工作时长'].agg([
        ('活塞工作时长','min'),('num_samples','count')])
    feat_df = feat_df.merge(fea, on=['sample_file_name'])
    
#     with timer(f'make corr feat'):
#         fea = df.groupby('sample_file_name').apply(apply_corr)
#         feat_df['发动机_油泵转速_corr'] = feat_df.sample_file_name.map(fea)
    
    fea = df.groupby('sample_file_name')['设备类型'].agg([
        ('设备类型',agg_first_value)]).reset_index()
    feat_df = feat_df.merge(fea, on=['sample_file_name'])
    
    device_mapper = {'ZV573': 0, 'ZVfd4': 1, 'ZVa9c': 2, 'ZV63d': 3, 'ZVa78': 4, 
                     'ZVe44': 5, 'ZV252': 6}
    feat_df['设备类型'] = feat_df['设备类型'].map(device_mapper)
    
    for col in ['发动机转速', '油泵转速', '泵送压力', 
                '液压油温', '流量档位', '分配压力', '排量电流']:
        with timer(f'make feature for {col}'):
            fea = df.groupby('sample_file_name')[col].agg([
                    (f'min_{col}','min'),
                    (f'max_{col}','max'),
                    # (f'median_{col}','median'),
                    (f'mean_{col}','mean'),
                    (f'nuni_{col}','nunique'),
                    # (f'std_{col}','std'),
                    # (f'skew_{col}','skew'),
                    # (f'kurtosis_{col}', agg_kurtosis),
                    # (f'normaltest_{col}',agg_normaltest_pvalue),
                    # (f'num_bins_{col}', agg_num_bins),
                    ]).reset_index()
        feat_df = feat_df.merge(fea, on=['sample_file_name'])
    
    for col in ['低压开关', '高压开关', '正泵', '反泵']:
        with timer(f'make feature for {col}'):
            fea = df.groupby('sample_file_name')[col].agg([
                (f'{col}',agg_first_value)]).reset_index()
        feat_df = feat_df.merge(fea, on=['sample_file_name'])
    
    return feat_df


def main(debug=False):
    NROWS = 200000 if debug else None
    with timer('load train_df'):
        train_df = pd.read_csv('../input/train_df.csv', nrows=NROWS)
    with timer('load test_df'):
        test_df = pd.read_csv('../input/test_df.csv', nrows=NROWS)
    with timer('make train_feat_df'):
        train_feat_df = make_features(train_df)
        label_df = pd.read_csv('../input/label_df.csv')
        train_feat_df = train_feat_df.merge(label_df)
    with timer('make test_feat_df'):
        test_feat_df = make_features(test_df)
    with timer('save train_feat_df'):
        train_feat_df.to_csv('../input/train_feat_df.csv',index=False)
    with timer('save test_feat_df'):
        test_feat_df.to_csv('../input/test_feat_df.csv',index=False)


if __name__ == '__main__':
    DEBUG = False
    main(DEBUG)