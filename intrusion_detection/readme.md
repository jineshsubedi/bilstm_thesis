# Data Collection:
```
    Original Class Distribution:
    type
    normal       300000
    backdoor      20000
    ddos          20000
    injection     20000
    dos           20000
    scanning      20000
    password      20000
    xss           20000
    mitm           1043
    Name: count, dtype: int64

    Original Label Distribution:
    label
    0       300000
    1       141043
    Name: count, dtype: int64
```
# Data Preprocessing:
-   Data Cleaning
-   Feature Engineering about TARGET_LABEL, TARGET_TYPE
-   Categorical Feature Encoding
-   Data Split:
    -   Initial Split
        -   Training Set = 352834
        -   Test Set = 88209
    -   Validation Split:
        -   Training Set = 285556
        -   Validation Set = 71389
-   MinMaxScalar is used after the data being split i.e X_train, X_test

```
    X_train shape: (352834, 44)
    y_train_multiclass labels shape: (352834, 9)
    X_test  shape: (88209, 44)
    y_test_multiclass labels shape: (88209, 9)
    Multi-class target class distribution (train):
    Counter({np.int64(5): 240000, np.int64(0): 16000, np.int64(6): 16000, np.int64(7): 16000, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(4): 834})
    Multi-class target class distribution (test):
    Counter({np.int64(5): 60000, np.int64(1): 4000, np.int64(3): 4000, np.int64(6): 4000, np.int64(0): 4000, np.int64(8): 4000, np.int64(7): 4000, np.int64(2): 4000, np.int64(4): 209})
```

-   Timestamp = 10
```
    Total sequences generated: 88200
    Train sequence shape: (352825, 10, 44)
    Test sequence shape: (88200, 10, 44)
    multi target distribution: Counter({np.int64(5): 240000, np.int64(0): 16000, np.int64(6): 16000, np.int64(7): 16000, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(4): 834})
    Class distribution in y_train_seq: Counter({np.int64(5): 239994, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(7): 15999, np.int64(6): 15999, np.int64(0): 15999, np.int64(4): 834})
    Class distribution in y_test_seq: Counter({np.int64(5): 59995, np.int64(8): 4000, np.int64(7): 4000, np.int64(2): 4000, np.int64(0): 3999, np.int64(1): 3999, np.int64(6): 3999, np.int64(3): 3999, np.int64(4): 209})
```
# Baseline Model:
## features
-   MODEL
    -   Bidirectional(LSTM = 64, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.25
    -   Bidirectional(LSTM = 32, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.25
    -   Dense(16, activation='relu', kernel_regularizer=l2(1e-4))

-   No SMOTE applied
-   loss='categorical_crossentropy'
-   Optimizer: Adam(learning_rate=0.0001, clipnorm=1.0)
-   No Class weights is used

## Report
-   CLASSIFICATION REPORT
```
    precision    recall  f1-score   support

            0       0.99      1.00      0.99      3999
            1       0.80      0.92      0.85      3999
            2       0.99      0.92      0.95      4000
            3       0.89      0.85      0.87      3999
            4       0.00      0.00      0.00       209
            5       0.99      0.98      0.98     59995
            6       0.93      0.89      0.91      3999
            7       0.97      0.97      0.97      4000
            8       0.88      0.95      0.91      4000

        accuracy                           0.96     88200
    macro avg       0.82      0.83      0.83     88200
    weighted avg       0.96      0.96      0.96     88200

    Class      FNR      FPR
    backdoor 0.000000 0.000570
        ddos 0.084771 0.010938
        dos 0.080000 0.000653
    injection 0.150288 0.005059
        mitm 1.000000 0.000000
    normal 0.016485 0.029427
    password 0.109027 0.003183
    scanning 0.028250 0.001651
        xss 0.049750 0.006164
```

# 1st Level Optimized Model:
## features
-   MODEL
    -   Bidirectional(LSTM = 128, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Bidirectional(LSTM = 64, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
    -   Dropout=0.25
    -   Dense(32, activation='relu')
    -   Dropout=0.25
    -   Dense(9, activation='softmax')

-   ADYSN is Applied for minority class(mitm = 5000)
-   loss=focal_loss(gamma=2.0, alpha=0.25)
-   Optimizer: Adam(learning_rate=0.0001, clipnorm=1.0)
-   For Class Weight, Median based weight calculation
-   f1_metric is used

## Report
-   CLASSIFICATION REPORT
```
    precision    recall  f1-score   support

            0       0.99      1.00      1.00      3999
            1       0.88      0.95      0.92      3999
            2       0.99      0.92      0.95      4000
            3       0.94      0.93      0.93      3999
            4       0.33      0.90      0.48       209
            5       0.99      0.98      0.98     59995
            6       0.92      0.81      0.86      3999
            7       0.94      0.99      0.96      4000
            8       0.83      0.98      0.90      4000

    accuracy                            0.96     88200
    macro avg       0.87      0.94      0.89     88200
    weighted avg    0.97      0.96      0.97     88200

        Class      FNR      FPR
        backdoor   0.000000 0.000416
            ddos   0.047512 0.006081
            dos    0.080000 0.000629
        injection  0.074019 0.003040
            mitm   0.095694 0.004455
        normal     0.022885 0.018401
        password   0.189797 0.003195
        scanning   0.014500 0.002743
            xss    0.020500 0.009869
```

# 2nd Level Optimized Model:
## features
-   MODEL
    -   Bidirectional(LSTM = 128, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Bidirectional(LSTM = 64, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
    -   Dropout=0.25
    -   Dense(32, activation='relu')
    -   Dropout=0.25
    -   Dense(9, activation='softmax')

-   ADYSN is Applied for minority class(mitm = 5000)
-   loss=focal_loss(gamma=2.0, alpha=0.25)
-   optimizer = Adam(learning_rate=CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=5), clipnorm=1.0)
-   For Class Weight, Median based weight calculation
-   GMeanMetric() is used
-   

## Report
-   CLASSIFICATION REPORT
```
    precision    recall  f1-score   support

           0       1.00      1.00      1.00      3999
           1       0.93      0.99      0.96      3999
           2       0.98      0.94      0.96      4000
           3       0.97      0.98      0.97      3999
           4       0.36      0.94      0.52       209
           5       0.99      0.98      0.99     59995
           6       0.97      1.00      0.98      3999
           7       0.98      1.00      0.99      4000
           8       0.98      0.98      0.98      4000

    accuracy                           0.98     88200
   macro avg       0.91      0.98      0.93     88200
weighted avg       0.99      0.98      0.98     88200

    Class      FNR      FPR
 backdoor 0.000000 0.000226
     ddos 0.010253 0.003290
      dos 0.056750 0.000701
injection 0.022756 0.001520
     mitm 0.062201 0.003921
   normal 0.016551 0.011913
 password 0.003501 0.001354
 scanning 0.004250 0.001116
      xss 0.015000 0.000998
```

# 3rd Level Optimized Model:
-   Dataset Update:
```
Original Class Distribution:
 type
normal       300000
backdoor      30000
ddos          30000
injection     30000
dos           30000
scanning      30000
password      30000
xss           30000
mitm           1043
Name: count, dtype: int64

Original Label Distribution:
 label
0    300000
1    211043
Name: count, dtype: int64

Final training shapes:
X_train_seq_final shape: (334324, 10, 44)
y_train_seq_final shape: (334324, 9)
X_val_seq shape: (83581, 10, 44)
y_val_seq shape: (83581, 9)
Final training label distributions:
y_train_seq_final distribution: Counter({np.int64(5): 192171, np.int64(2): 19243, np.int64(3): 19219, np.int64(0): 19205, np.int64(6): 19170, np.int64(1): 19153, np.int64(8): 19139, np.int64(7): 19130, np.int64(4): 7894})
y_val_seq distribution: Counter({np.int64(5): 47821, np.int64(7): 4870, np.int64(8): 4861, np.int64(1): 4846, np.int64(6): 4830, np.int64(0): 4795, np.int64(3): 4781, np.int64(2): 4757, np.int64(4): 2020})


```

## features
-   MODEL
    -   Bidirectional(LSTM = 128, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Bidirectional(LSTM = 64, return_sequences=True, recurrent_dropout=0.1,recurrent_activation='sigmoid',activation='tanh',kernel_regularizer=l2(1e-4))
    -   LayerNormalization
    -   Dropout=0.3
    -   Dense(64, activation='relu', kernel_regularizer=l2(1e-4))
    -   Dropout=0.25
    -   Dense(32, activation='relu')
    -   Dropout=0.25
    -   Dense(9, activation='softmax')
-   ADYSN is Applied for minority class(mitm = 10000)
-   loss=focal_loss(gamma=2.0, alpha=0.25)
-   optimizer = Adam(learning_rate=CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=5), clipnorm=1.0)
-   For Class Weight, Median based weight calculation
-   GMeanMetric() is used


## Report
-   CLASSIFICATION REPORT
```
    precision    recall  f1-score   support

           0       0.99      1.00      1.00      6000
           1       0.95      0.99      0.97      5999
           2       0.97      0.96      0.97      5999
           3       0.96      0.99      0.98      5999
           4       0.31      0.95      0.47       209
           5       1.00      0.98      0.99     59995
           6       0.96      1.00      0.97      6000
           7       0.98      1.00      0.99      6000
           8       0.98      0.97      0.98      5999

    accuracy                           0.98    102200
   macro avg       0.90      0.98      0.92    102200
weighted avg       0.98      0.98      0.98    102200

    Class      FNR      FPR
 backdoor 0.000333 0.000364
     ddos 0.013002 0.003430
      dos 0.037673 0.001601
injection 0.005001 0.002692
     mitm 0.047847 0.004304
   normal 0.024352 0.006587
 password 0.004833 0.002900
 scanning 0.004000 0.001227
      xss 0.025004 0.001227
```
