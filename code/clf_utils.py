import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from contextlib import contextmanager
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import warnings
from scipy.stats import pearsonr
import seaborn as sns
# import tsfresh.feature_extraction.feature_calculators as feature_calculator
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

def metric_micro_f1(y_true, y_pred):
    return 'f1', f1_score(y_true, y_pred>0.5, average="micro"), True

def xgb_metric_micro_f1(y_pred, y_true):
    y_true = y_true.get_label()
    name, score = metric_micro_f1(y_true, y_pred)[:2]
    return name, -score


def kf_lgbm(x, y, x_test, output_dir='.', objective="binary", eval_metric=metric_micro_f1, 
            name="lgb", 
            n_estimators=3000,
            early_stopping_rounds=50, 
            learning_rate=0.01, 
            num_leaves=31, 
            max_depth=-1,
            max_bin=255, 
            reg_alpha=0.0, 
            reg_lambda=1.0, 
            colsample_bytree=0.5, 
            subsample=0.8,
            subsample_freq=2, 
            min_split_gain=1,
            min_child_samples=20, 
            verbose=200,
            n_folds=10, 
            split_seed=8888, 
            boosting_type="gbdt",
            categorical_feature=['设备类型'], **kwargs):
    oof_train = np.zeros(x.shape[0])
    oof_test = np.zeros(x_test.shape[0])
    fold_idx = 1
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    model = None
    for train_idx, valid_idx in kf.split(x, y):
        print()
        print("=" * 25, "Fold %d" % fold_idx, "=" * 25)
        fold_idx += 1
        if isinstance(x, pd.DataFrame):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        else:
            x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = lgb.LGBMRegressor(boosting_type=boosting_type,
                                  learning_rate=learning_rate,
                                  num_leaves=num_leaves,
                                  max_depth=max_depth,
                                  n_estimators=n_estimators,
                                  max_bin=max_bin,
                                  objective=objective,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda,
                                  colsample_bytree=colsample_bytree,
                                  subsample=subsample,
                                  subsample_freq=subsample_freq,
                                  min_child_samples=min_child_samples,
                                  min_split_gain=min_split_gain,
                                  metric="auc",
                                  **kwargs)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train),
                                              (x_valid, y_valid)],
                  eval_names=['train', 'test'], eval_metric=eval_metric,
                  verbose=verbose, early_stopping_rounds=early_stopping_rounds,
                  categorical_feature=categorical_feature)
        oof_train[valid_idx] = model.predict(x_valid,num_iteration=model.best_iteration_)
        oof_test += model.predict(x_test, num_iteration=model.best_iteration_)
    oof_test /= n_folds
    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), oof_test)
    return model

def kf_xgbm(x, y, x_test, output_dir='.', objective="binary:logistic", 
            eval_metric=xgb_metric_micro_f1, 
            name="xgb", 
            n_estimators=3000,
            early_stopping_rounds=50, 
            learning_rate=0.01, 
            max_depth=6,
            reg_alpha=0.0, 
            reg_lambda=1.0, 
            colsample_bytree=0.5, 
            subsample=0.8,
            verbose=200,
            n_folds=10, 
            split_seed=8888,
            **kwargs):
    oof_train = np.zeros(x.shape[0])
    oof_test = np.zeros(x_test.shape[0])
    fold_idx = 1
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    model = None
    for train_idx, valid_idx in kf.split(x, y):
        print()
        print("=" * 25, "Fold %d" % fold_idx, "=" * 25)
        fold_idx += 1
        if isinstance(x, pd.DataFrame):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        else:
            x_train, x_valid = x[train_idx], x[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = xgb.XGBClassifier(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  n_estimators=n_estimators,
                                  objective=objective,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda,
                                  colsample_bytree=colsample_bytree,
                                  subsample=subsample,
                                  eval_metric="auc",
                                  **kwargs)
        model.fit(x_train, y_train, eval_set=[(x_train, y_train),
                                              (x_valid, y_valid)],
                  eval_metric=eval_metric,verbose=verbose, 
                  early_stopping_rounds=early_stopping_rounds)

        oof_train[valid_idx] = model.predict_proba(x_valid,ntree_limit=model.best_iteration)[:,1]
        oof_test += model.predict_proba(x_test, ntree_limit=model.best_iteration)[:,1]
    oof_test /= n_folds
    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), oof_test)
    return model


def kf_sklearn(x, y, x_test, output_dir='./rf/',model_class=RandomForestClassifier,
            name="rf", n_folds=10, split_seed=8888, **kwargs):
    oof_train = np.zeros(x.shape[0])
    oof_test = np.zeros(x_test.shape[0])
    fold_idx = 1
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    model = None
    for train_idx, valid_idx in kf.split(x, y):
        print()
        print("=" * 25, "Fold %d" % fold_idx, "=" * 25)
        fold_idx += 1
        if isinstance(x, pd.DataFrame):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        else:
            x_train, x_valid = x[train_idx], x[valid_idx]        
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = model_class(**kwargs)
        model.fit(x_train, y_train)
        oof_train[valid_idx] = model.predict_proba(x_valid)[:,1]
        oof_test += model.predict_proba(x_test)[:,1]
    oof_test /= n_folds
    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), oof_test)
    return model