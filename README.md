# practice_xgboost_prediction

### 1.How to predict 999901_2017M06?

1. Change param. Exec 01. Generate 3 train feature and 1 text feature。

| data_time | feature_time | isTrain | type       |
| --------- | ------------ | ------- | ---------- |
| 2015_12   | 2016_09       | True    | notMissing |
| 2016_03   | 2016_12       | True    | notMissing |
| 2016_06   | 2017_03       | True    | notMissing |
| 2016_09   | 2017_06       | False    | missing |

2. Change param. Exec 02. Train model and select best model‘s param

   |train_time|
   | --------- |
   |2016_09|

3. Change param. Exec 03. Use test feature predict 2017M06.

   

   | train_time | test_year |
   | ---------- | --------- |
   | 2016_09    | 2017_06   |

4.  Exec 04. Compare xgboost's result with GDG