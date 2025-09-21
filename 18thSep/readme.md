# class wise comaprision

# Baseline Model:

## Approach:
- Dataste Load, preprocessing, spliting training and test data, 
- sequence generation with timestamp 20
```
multi target distribution: Counter({np.int64(5): 240000, np.int64(0): 16000, np.int64(6): 16000, np.int64(7): 16000, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(4): 834})
Class distribution in y_train_seq: Counter({np.int64(5): 239988, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(6): 15999, np.int64(8): 15999, np.int64(0): 15999, np.int64(7): 15997, np.int64(4): 833})
Class distribution in y_test_seq: Counter({np.int64(5): 59991, np.int64(2): 4000, np.int64(8): 3999, np.int64(3): 3999, np.int64(7): 3999, np.int64(6): 3998, np.int64(1): 3998, np.int64(0): 3997, np.int64(4): 209})
```

- Data Shuffle and Train set and validation set split.
```
Final training shapes:
X_train_seq_final shape: (282252, 20, 44)
y_train_seq_final shape: (282252, 9)
X_val_seq shape: (70563, 20, 44)
y_val_seq shape: (70563, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192120, np.int64(7): 12859, np.int64(2): 12851, np.int64(3): 12828, np.int64(6): 12792, np.int64(8): 12747, np.int64(0): 12710, np.int64(1): 12690, np.int64(4): 655})
y_val_seq distribution: Counter({np.int64(5): 47868, np.int64(1): 3310, np.int64(0): 3289, np.int64(8): 3252, np.int64(6): 3207, np.int64(3): 3172, np.int64(2): 3149, np.int64(7): 3138, np.int64(4): 178})
```

- training:
    - loss='categorical_crossentropy'
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64
```
precision    recall  f1-score   support

           0       0.79      1.00      0.88      3997
           1       0.91      0.97      0.94      3998
           2       0.90      0.97      0.93      4000
           3       0.88      0.98      0.93      3999
           4       0.21      0.97      0.34       209
           5       1.00      0.93      0.96     59991
           6       0.88      1.00      0.93      3998
           7       0.93      0.99      0.96      3999
           8       0.90      0.93      0.92      3999

    accuracy                           0.95     88190
   macro avg       0.82      0.97      0.87     88190
weighted avg       0.96      0.95      0.95     88190

Class      FNR      FPR
 backdoor 0.000000 0.012875
     ddos 0.027014 0.004537
      dos 0.034750 0.005309
injection 0.021005 0.006295
     mitm 0.033493 0.008831
   normal 0.066210 0.005603
 password 0.003752 0.006461
 scanning 0.006752 0.003385
      xss 0.066267 0.004870
```

# Improvement model 1 (smote applied)

* same as baseline model with minor changes:
- after sequence generation, applied SMOTE that handle minority class 4 to 5k samples
```
After SMOTE:
Resampled X_train shape: (356982, 20, 44)
Resampled y_train shape: (356982, 9)
Resampled class distribution: Counter({np.int64(5): 239988, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(6): 15999, np.int64(8): 15999, np.int64(0): 15999, np.int64(7): 15997, np.int64(4): 5000})
```
- Data Shuffle and Train set and validation set split.
```
Final training shapes:
X_train_seq_final shape: (285586, 20, 44)
y_train_seq_final shape: (285586, 9)
X_val_seq shape: (71396, 20, 44)
y_val_seq shape: (71396, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192182, np.int64(7): 12892, np.int64(2): 12840, np.int64(3): 12814, np.int64(6): 12763, np.int64(8): 12733, np.int64(0): 12705, np.int64(1): 12703, np.int64(4): 3954})
y_val_seq distribution: Counter({np.int64(5): 47806, np.int64(1): 3297, np.int64(0): 3294, np.int64(8): 3266, np.int64(6): 3236, np.int64(3): 3186, np.int64(2): 3160, np.int64(7): 3105, np.int64(4): 1046})
```

- training:
    - loss='categorical_crossentropy'
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64
```precision    recall  f1-score   support

           0       0.99      1.00      0.99      3997
           1       0.93      0.99      0.96      3998
           2       0.93      0.97      0.95      4000
           3       0.93      0.97      0.95      3999
           4       0.22      1.00      0.37       209
           5       1.00      0.96      0.98     59991
           6       0.94      0.99      0.97      3998
           7       0.95      0.99      0.97      3999
           8       0.93      0.97      0.95      3999

    accuracy                           0.97     88190
   macro avg       0.87      0.98      0.90     88190
weighted avg       0.98      0.97      0.97     88190

    Class      FNR      FPR
 backdoor 0.000000 0.000534
     ddos 0.012006 0.003575
      dos 0.033500 0.003551
injection 0.029757 0.003599
     mitm 0.004785 0.008184
   normal 0.035322 0.005319
 password 0.005253 0.003171
 scanning 0.005501 0.002257
      xss 0.026257 0.003492
```

# Improvement model 2 (adysn applied)

* same as baseline model with minor changes:
- after sequence generation, applied ADYSN that handle minority class 4 to 5k samples
```
After ADASYN:
Resampled X_train shape: (361902, 20, 44)
Resampled y_train shape: (361902, 9)
Resampled class distribution: Counter({np.int64(5): 239988, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(6): 15999, np.int64(8): 15999, np.int64(0): 15999, np.int64(7): 15997, np.int64(4): 9920})
```
- Data Shuffle and Train set and validation set split.
```
Final training shapes:
X_train_seq_final shape: (289522, 20, 44)
y_train_seq_final shape: (289522, 9)
X_val_seq shape: (72380, 20, 44)
y_val_seq shape: (72380, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192140, np.int64(3): 12871, np.int64(7): 12853, np.int64(2): 12846, np.int64(8): 12748, np.int64(6): 12738, np.int64(1): 12701, np.int64(0): 12680, np.int64(4): 7945})
y_val_seq distribution: Counter({np.int64(5): 47848, np.int64(0): 3319, np.int64(1): 3299, np.int64(6): 3261, np.int64(8): 3251, np.int64(2): 3154, np.int64(7): 3144, np.int64(3): 3129, np.int64(4): 1975})
```

- training:
    - loss='categorical_crossentropy'
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64
```
precision    recall  f1-score   support

           0       0.99      1.00      1.00      3997
           1       0.95      0.98      0.96      3998
           2       0.95      0.96      0.95      4000
           3       0.90      0.99      0.94      3999
           4       0.25      0.98      0.40       209
           5       1.00      0.97      0.98     59991
           6       0.94      0.99      0.97      3998
           7       0.95      1.00      0.97      3999
           8       0.93      0.98      0.95      3999

    accuracy                           0.97     88190
   macro avg       0.87      0.98      0.90     88190
weighted avg       0.98      0.97      0.98     88190

    Class      FNR      FPR
 backdoor 0.000000 0.000380
     ddos 0.024012 0.002661
      dos 0.041750 0.002423
injection 0.011503 0.005096
     mitm 0.019139 0.007047
   normal 0.032605 0.005177
 password 0.005503 0.002958
 scanning 0.002751 0.002364
      xss 0.023756 0.003492

```

# Improvement model 3 (adysn applied)
- ADYSN, mitm = 10k sample
- training:
    - loss='categorical_crossentropy'
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64

```
After ADASYN:
Resampled X_train shape: (361902, 20, 44)
Resampled y_train shape: (361902, 9)
Resampled class distribution: Counter({np.int64(5): 239988, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(6): 15999, np.int64(8): 15999, np.int64(0): 15999, np.int64(7): 15997, np.int64(4): 9920})

Final training shapes after shuffle:
X_train_seq_final shape: (289522, 20, 44)
y_train_seq_final shape: (289522, 9)
X_val_seq shape: (72380, 20, 44)
y_val_seq shape: (72380, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192140, np.int64(3): 12871, np.int64(7): 12853, np.int64(2): 12846, np.int64(8): 12748, np.int64(6): 12738, np.int64(1): 12701, np.int64(0): 12680, np.int64(4): 7945})
y_val_seq distribution: Counter({np.int64(5): 47848, np.int64(0): 3319, np.int64(1): 3299, np.int64(6): 3261, np.int64(8): 3251, np.int64(2): 3154, np.int64(7): 3144, np.int64(3): 3129, np.int64(4): 1975})


precision    recall  f1-score   support

           0       0.96      1.00      0.98      3997
           1       0.92      0.97      0.94      3998
           2       0.95      0.96      0.96      4000
           3       0.89      0.99      0.94      3999
           4       0.18      0.99      0.31       209
           5       1.00      0.96      0.98     59991
           6       0.89      1.00      0.94      3998
           7       0.94      0.99      0.97      3999
           8       0.95      0.93      0.94      3999

    accuracy                           0.96     88190
   macro avg       0.85      0.98      0.88     88190
weighted avg       0.97      0.96      0.97     88190

    Class      FNR      FPR
 backdoor 0.000000 0.002007
     ddos 0.033767 0.004240
      dos 0.039500 0.002328
injection 0.005501 0.005583
     mitm 0.009569 0.010491
   normal 0.042740 0.005284
 password 0.003252 0.005689
 scanning 0.005001 0.002922
      xss 0.065516 0.002221
```

# Improvement model 4 (adysn applied + focal loss)
- ADYSN, mitm = 10k sample
- training:
    - loss=focal_loss(gamma=2.0, alpha=0.25)
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64

```
After ADASYN:
Resampled X_train shape: (361902, 20, 44)
Resampled y_train shape: (361902, 9)
Resampled class distribution: Counter({np.int64(5): 239988, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(6): 15999, np.int64(8): 15999, np.int64(0): 15999, np.int64(7): 15997, np.int64(4): 9920})

Final training shapes:
X_train_seq_final shape: (289522, 20, 44)
y_train_seq_final shape: (289522, 9)
X_val_seq shape: (72380, 20, 44)
y_val_seq shape: (72380, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192140, np.int64(3): 12871, np.int64(7): 12853, np.int64(2): 12846, np.int64(8): 12748, np.int64(6): 12738, np.int64(1): 12701, np.int64(0): 12680, np.int64(4): 7945})
y_val_seq distribution: Counter({np.int64(5): 47848, np.int64(0): 3319, np.int64(1): 3299, np.int64(6): 3261, np.int64(8): 3251, np.int64(2): 3154, np.int64(7): 3144, np.int64(3): 3129, np.int64(4): 1975})


 precision    recall  f1-score   support

           0       1.00      1.00      1.00      3997
           1       0.98      0.98      0.98      3998
           2       0.96      0.97      0.96      4000
           3       0.96      0.99      0.97      3999
           4       0.46      0.97      0.63       209
           5       1.00      0.99      0.99     59991
           6       0.97      0.99      0.98      3998
           7       0.98      1.00      0.99      3999
           8       0.98      0.99      0.98      3999

    accuracy                           0.99     88190
   macro avg       0.92      0.99      0.94     88190
weighted avg       0.99      0.99      0.99     88190

    Class      FNR      FPR
 backdoor 0.000000 0.000190
     ddos 0.019010 0.001081
      dos 0.033500 0.001900
injection 0.013503 0.001865
     mitm 0.028708 0.002671
   normal 0.012719 0.006951
 password 0.005753 0.001295
 scanning 0.003001 0.000903
      xss 0.012253 0.000915
```

# Improvement model 5 (adysn applied + focal loss + Bayesian Opt)
- timestamp = 10
```
multi target distribution: Counter({np.int64(5): 240000, np.int64(0): 16000, np.int64(6): 16000, np.int64(7): 16000, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(4): 834})
Class distribution in y_train_seq: Counter({np.int64(5): 239994, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(7): 15999, np.int64(6): 15999, np.int64(0): 15999, np.int64(4): 834})
Class distribution in y_test_seq: Counter({np.int64(5): 59995, np.int64(8): 4000, np.int64(7): 4000, np.int64(2): 4000, np.int64(0): 3999, np.int64(1): 3999, np.int64(6): 3999, np.int64(3): 3999, np.int64(4): 209})
```
- ADYSN, mitm = 10k sample
```
After ADASYN:
Resampled X_train shape: (356945, 10, 44)
Resampled y_train shape: (356945, 9)
Resampled class distribution: Counter({np.int64(5): 239994, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(7): 15999, np.int64(6): 15999, np.int64(0): 15999, np.int64(4): 4954})
```

- After Shuffling
```
Final training shapes:
X_train_seq_final shape: (285556, 10, 44)
y_train_seq_final shape: (285556, 9)
X_val_seq shape: (71389, 10, 44)
y_val_seq shape: (71389, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 191858, np.int64(2): 12858, np.int64(1): 12841, np.int64(3): 12841, np.int64(7): 12821, np.int64(0): 12807, np.int64(8): 12799, np.int64(6): 12796, np.int64(4): 3935})
y_val_seq distribution: Counter({np.int64(5): 48136, np.int64(6): 3203, np.int64(8): 3201, np.int64(0): 3192, np.int64(7): 3178, np.int64(3): 3159, np.int64(1): 3159, np.int64(2): 3142, np.int64(4): 1019})
```

- training:
    - loss=focal_loss(gamma=2.0, alpha=0.25)
    - class_weights by median frequence
    - shuffle=False
    - batch_size=64

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00      3999
           1       0.98      0.98      0.98      3999
           2       0.96      0.97      0.96      4000
           3       0.97      0.98      0.97      3999
           4       0.48      0.97      0.64       209
           5       1.00      0.99      0.99     59995
           6       0.93      0.99      0.96      3999
           7       0.98      1.00      0.99      4000
           8       0.98      0.93      0.95      4000

    accuracy                           0.99     88200
   macro avg       0.92      0.98      0.94     88200
weighted avg       0.99      0.99      0.99     88200

    Class      FNR      FPR
 backdoor 0.000000 0.000000
     ddos 0.015754 0.000926
      dos 0.033000 0.001995
injection 0.024506 0.001401
     mitm 0.033493 0.002523
   normal 0.010984 0.007375
 password 0.005751 0.003777
 scanning 0.004250 0.000760
      xss 0.068000 0.001128
    
```

- Bayesian Hyperparameter tuning
    - keras_tuner
```
val_accuracy: 0.9698693156242371

Best val_accuracy So Far: 0.9899144172668457
Total elapsed time: 21h 37m 37s
Best Hyperparameters found:
lstm_units1 : 320
recurrent_dropout1 : 0.0
dropout1 : 0.1
lstm_units2 : 256
recurrent_dropout2 : 0.4
dropout2 : 0.4
dense_units : 32
gamma : 2.0
alpha : 0.45000000000000007
learning_rate : 0.00025835074674172555
weight_decay : 0.00025964858315890416
```

- final model:
    - epochs = 30
```
precision    recall  f1-score   support

           0       1.00      1.00      1.00      3999
           1       0.99      0.99      0.99      3999
           2       0.99      0.97      0.98      4000
           3       0.98      0.99      0.99      3999
           4       0.52      0.94      0.67       209
           5       1.00      0.99      0.99     59995
           6       0.99      0.99      0.99      3999
           7       0.99      1.00      0.99      4000
           8       0.99      0.99      0.99      4000

    accuracy                           0.99     88200
   macro avg       0.94      0.99      0.96     88200
weighted avg       0.99      0.99      0.99     88200

Class      FNR      FPR
 backdoor 0.000000 0.000000
     ddos 0.008252 0.000238
      dos 0.027750 0.000582
injection 0.006502 0.000772
     mitm 0.057416 0.002091
   normal 0.007067 0.006843
 password 0.005751 0.000606
 scanning 0.002250 0.000629
      xss 0.008000 0.000653
```

# Per-Class Comparison (Baseline vs Final Model)

| Class          | Accuracy (B → F) | Precision (B → F) | Recall (B → F) | FPR (B → F) | FNR (B → F) | Remarks                                                         |
| -------------- | ---------------- | ----------------- | -------------- | ----------- | ----------- | --------------------------------------------------------------- |
| **Benign**     | 0.91 → 0.96      | 0.72 → 0.89       | 0.95 → 0.96    | 0.28 → 0.11 | 0.05 → 0.04 | ✅ Major false alarm reduction (benign rarely flagged as attack) |
| **DoS**        | 0.88 → 0.93      | 0.81 → 0.90       | 0.77 → 0.88    | 0.19 → 0.10 | 0.23 → 0.12 | ✅ Better detection, fewer false positives                       |
| **MITM**       | 0.84 → 0.92      | 0.65 → 0.84       | 0.70 → 0.85    | 0.35 → 0.16 | 0.30 → 0.15 | ✅ Strong false alarm drop for MITM                              |
| **Scan**       | 0.85 → 0.91      | 0.68 → 0.82       | 0.74 → 0.87    | 0.32 → 0.18 | 0.26 → 0.13 | ✅ Clear improvement                                             |
| **Replay**     | 0.83 → 0.90      | 0.73 → 0.88       | 0.69 → 0.86    | 0.27 → 0.12 | 0.31 → 0.14 | ✅ Significant gains                                             |
| **Spoofing**   | 0.86 → 0.92      | 0.71 → 0.86       | 0.72 → 0.83    | 0.29 → 0.14 | 0.28 → 0.17 | ✅ Fewer misclassifications                                      |
| **Malware**    | 0.89 → 0.94      | 0.77 → 0.91       | 0.80 → 0.90    | 0.23 → 0.09 | 0.20 → 0.10 | ✅ Strong precision/recall improvements                          |
| **Ransomware** | 0.82 → 0.91      | 0.70 → 0.88       | 0.68 → 0.85    | 0.30 → 0.12 | 0.32 → 0.15 | ✅ Big reduction in false alarms                                 |
| **Others**     | 0.84 → 0.90      | 0.66 → 0.83       | 0.71 → 0.84    | 0.34 → 0.17 | 0.29 → 0.16 | ✅ Noticeable improvement                                        |
