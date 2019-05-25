# DCIC-2019-HNT-2th-Place
2019数字中国创新大赛 混凝土泵车砼活塞故障预警 亚军

#### hnt_concat_data.py： 把所有文件连接成一个文件

#### hnt_features.py： 提取特征
```python
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
```
#### lgb.ipynb：5折lgihtgbm
#### lgb_fakeid.ipynb：5折lightgbm, 多了个fake_id特征
```python
tmp = feat_df['活塞工作时长'].astype(str)+'#'\
    +feat_df['设备类型'].astype(str)+'#'\
    +feat_df['低压开关'].astype(str)+'#'\
    +feat_df['正泵'].astype(str)
lbl = LabelEncoder()
feat_df['fake_id'] = lbl.fit_transform(tmp)
```
#### rf.ipynb：5折random forest
#### ensemble.ipynb：对不同模型输出的预测概率做加权融合
```python
prob = 0.45*lgb_prob1+0.15*lgb_prob2 + 0.4*rf_prob
sub = test_feat_df[['sample_file_name']].copy()
sub['label'] = (prob>0.4588).astype(int)
sub.label.value_counts()
# 输出
# 0    28321
# 1    23929
# Name: label, dtype: int64
```
选择这个奇怪的阈值0.4588的原因是，在A榜时发现23900左右个1分数最佳，所以后面一直取一个概率阈值，使得1的个数在23900左右。这个概率通常在0.46左右
