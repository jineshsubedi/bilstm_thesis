# IDS Research

## Test 1 - ids_1.ipynb

### steps performed
    * Data Import
    * Data Preprocessing

    ```
    Multi-class target class distribution:
    Counter({np.int64(5): 300000, np.int64(0): 20000, np.int64(1): 20000, np.int64(2): 20000, np.int64(3): 20000, np.int64(6): 20000, np.int64(7): 20000, np.int64(8): 20000, np.int64(4): 1043})

    ```
    * Data Split Training and Test by the test_size = 0.2
    ```
    Multi-class target class distribution (train):
    Counter({np.int64(5): 240000, np.int64(0): 16000, np.int64(6): 16000, np.int64(7): 16000, np.int64(8): 16000, np.int64(2): 16000, np.int64(1): 16000, np.int64(3): 16000, np.int64(4): 834})
    Multi-class target class distribution (test):
    Counter({np.int64(5): 60000, np.int64(1): 4000, np.int64(3): 4000, np.int64(6): 4000, np.int64(0): 4000, np.int64(8): 4000, np.int64(7): 4000, np.int64(2): 4000, np.int64(4): 209})
    Scaling complete.
    ```
    * Again split Training Data into Train Data and Validation Data
    ```
    Multi-class target class distribution (train):
    Counter({np.int64(5): 192000, np.int64(1): 12800, np.int64(3): 12800, np.int64(2): 12800, np.int64(7): 12800, np.int64(0): 12800, np.int64(8): 12800, np.int64(6): 12800, np.int64(4): 667})
    Multi-class target class distribution (val):
    Counter({np.int64(5): 48000, np.int64(6): 3200, np.int64(8): 3200, np.int64(0): 3200, np.int64(7): 3200, np.int64(1): 3200, np.int64(2): 3200, np.int64(3): 3200, np.int64(4): 167})
    ```
    * SMOTE only class 4 upto 5k
    ```
    X_train_res shape: (286600, 44)
    y_train_multi_res shape: (286600, 9)
    Class distribution (resampled train): Counter({np.int64(5): 192000, np.int64(1): 12800, np.int64(3): 12800, np.int64(2): 12800, np.int64(7): 12800, np.int64(0): 12800, np.int64(8): 12800, np.int64(6): 12800, np.int64(4): 5000})
    ```
    * Sequence Generation
    ```
    multi target distribution: Counter({np.int64(5): 192000, np.int64(1): 12800, np.int64(3): 12800, np.int64(2): 12800, np.int64(7): 12800, np.int64(0): 12800, np.int64(8): 12800, np.int64(6): 12800, np.int64(4): 5000})
    Class distribution in y_train_seq: Counter({np.int64(5): 191990, np.int64(6): 12800, np.int64(0): 12799, np.int64(8): 12799, np.int64(1): 12799, np.int64(2): 12798, np.int64(3): 12798, np.int64(7): 12798, np.int64(4): 5000})
    Class distribution in y_val_multi_seq: Counter({np.int64(5): 47989, np.int64(1): 3200, np.int64(2): 3200, np.int64(3): 3200, np.int64(0): 3199, np.int64(7): 3198, np.int64(8): 3198, np.int64(6): 3197, np.int64(4): 167})
    Class distribution in y_test_seq: Counter({np.int64(5): 59991, np.int64(2): 4000, np.int64(8): 3999, np.int64(3): 3999, np.int64(7): 3999, np.int64(6): 3998, np.int64(1): 3998, np.int64(0): 3997, np.int64(4): 209})
    ```
    * Model Training

    ```
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ bidirectional (Bidirectional)   │ (None, 20, 256)        │       177,152 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ layer_normalization             │ (None, 20, 256)        │           512 │
    │ (LayerNormalization)            │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout (Dropout)               │ (None, 20, 256)        │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ bidirectional_1 (Bidirectional) │ (None, 128)            │       164,352 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ layer_normalization_1           │ (None, 128)            │           256 │
    │ (LayerNormalization)            │                        │               │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dropout_1 (Dropout)             │ (None, 128)            │             0 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense (Dense)                   │ (None, 64)             │         8,256 │
    ├─────────────────────────────────┼────────────────────────┼───────────────┤
    │ dense_1 (Dense)                 │ (None, 9)              │           585 │
    └─────────────────────────────────┴────────────────────────┴───────────────┘
    Total Params: 351,113 (1.34 MB)
    Trainable params: 351,113 (1.34 MB)
    Non-trainable params: 0 (0.00 B)

    ```
    * Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3, 
            mode='min',
            restore_best_weights=True,
            verbose=1
        )

    * Reduce learning rate after 2 epochs of no improvement
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )

    * Class Weights (use Median Approach)
        frequencies = np.sum(y_train_seq, axis=0)
        median_freq = np.median(frequencies)
        class_weights = {cls: median_freq / count for cls, count in enumerate(frequencies)}

    ```
    Class weights for training: {0: np.float64(1.0), 1: np.float64(1.0), 2: np.float64(1.000078137208939), 3: np.float64(1.000078137208939), 4: np.float64(2.5598), 5: np.float64(0.06666493046512839), 6: np.float64(0.999921875), 7: np.float64(1.000078137208939), 8: np.float64(1.0)}

    ```
    
