import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t = time.time()
    yield
    print('[%s] done in %.2f seconds' % (name, time.time() - t))


def load_df(input_dir):
    filenames = os.listdir(input_dir)
    df_list = []
    for filename in tqdm(filenames):
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath)
        df['sample_file_name'] = filename
        df_list.append(df)
    df = pd.concat(df_list)
    return df


def main():
    with timer('load train_df'):
        train_df = load_df('../input/data_train/')
    with timer('save train_df'):
        train_df.to_csv('../input/train_df.csv',index=False)
    with timer('load test_df'):
        test_df = load_df('../input/data_test2/')
    with timer('save test_df'):
        test_df.to_csv('../input/test_df.csv',index=False)
    label_df = pd.read_csv('../input/train_labels.csv')
    label_df.to_csv('../input/label_df.csv',index=False)
    print(train_df['设备类型'].unique())
    print(test_df['设备类型'].unique())


if __name__ == '__main__':
    main()