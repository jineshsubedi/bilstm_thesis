# class wise comaprision

| Class             | No Sampling (P / R / F1) | SMOTE (P / R / F1) | ADASYN (P / R / F1)    |
| ----------------- | ------------------------ | ------------------ | ---------------------- |
| **0 (backdoor)**  | 0.79 / 1.00 / 0.88       | 0.99 / 1.00 / 0.99 | 0.99 / 1.00 / 1.00     |
| **1 (ddos)**      | 0.91 / 0.97 / 0.94       | 0.93 / 0.99 / 0.96 | 0.95 / 0.98 / 0.96     |
| **2 (dos)**       | 0.90 / 0.97 / 0.93       | 0.93 / 0.97 / 0.95 | 0.95 / 0.96 / 0.95     |
| **3 (injection)** | 0.88 / 0.98 / 0.93       | 0.93 / 0.97 / 0.95 | 0.90 / 0.99 / 0.94     |
| **4 (mitm)**      | 0.21 / 0.97 / 0.34       | 0.22 / 1.00 / 0.37 | **0.25 / 0.98 / 0.40** |
| **5 (normal)**    | 1.00 / 0.93 / 0.96       | 1.00 / 0.96 / 0.98 | 1.00 / 0.97 / 0.98     |
| **6 (password)**  | 0.88 / 1.00 / 0.93       | 0.94 / 0.99 / 0.97 | 0.94 / 0.99 / 0.97     |
| **7 (scanning)**  | 0.93 / 0.99 / 0.96       | 0.95 / 0.99 / 0.97 | 0.95 / 1.00 / 0.97     |
| **8 (xss)**       | 0.90 / 0.93 / 0.92       | 0.93 / 0.97 / 0.95 | 0.93 / 0.98 / 0.95     |


# Overall Comparision

| Metric              | No Sampling | SMOTE | ADASYN   |
| ------------------- | ----------- | ----- | -------- |
| **Accuracy**        | 0.95        | 0.97  | 0.97     |
| **Macro Precision** | 0.82        | 0.87  | 0.87     |
| **Macro Recall**    | 0.97        | 0.98  | 0.98     |
| **Macro F1**        | 0.87        | 0.90  | 0.90     |
| **Weighted F1**     | 0.95        | 0.97  | **0.98** |

# Analysis (ready to paste into thesis/report)

-   No Sampling:
    -   Minority class (mitm) suffers heavily with low precision (0.21) and low F1 (0.34).
    -   Several majority classes also show weaker performance (backdoor F1 = 0.88, dos = 0.93).
    -   Overall accuracy is 0.95, but imbalance reduces fairness across classes.
-   SMOTE:
    -   Strong recall across all classes, especially mitm with recall = 1.00 (no missed attacks).
    -   Precision remains low for mitm (0.22), leading to more false positives.
    -   Improves macro/weighted F1 to 0.90 / 0.97.
    -   Best for maximizing detection (recall) in intrusion detection.
-   ADASYN:
    -   Improves minority class F1 to 0.40 (better than SMOTE).
    -   Precision slightly higher (0.25), recall still strong (0.98).
    -   Weighted F1 0.98 = best overall balance across classes.
    -   Best for reducing false alarms while maintaining high recall.


# use of focal loss effectively addresses class imbalance and improves detection of rare yet important attacks. [ids_multi_adysn_median_weight_focalloss.ipynb]

| Metric                   | Model 1 (Cross-Entropy) | Model 2 (Focal Loss)     | Verdict      |
| ------------------------ | ----------------------- | ------------------------ | ------------ |
| Accuracy                 | 97%                     | 99%                      | Model 2 wins |
| Macro F1                 | 0.90                    | 0.94                     | Model 2 wins |
| Weighted F1              | 0.98                    | 0.99                     | Model 2 wins |
| Minority class detection | Poor (Class 4 F1=0.40)  | Better (Class 4 F1=0.63) | Model 2 wins |
| False Negative Reduction | Moderate                | Significant              | Model 2 wins |
| False Positive Reduction | Moderate                | Better                   | Model 2 wins |

| Class     | Model 1 FNR | Model 2 FNR | Improvement                    |
| --------- | ----------- | ----------- | ------------------------------ |
| backdoor  | 0.00        | 0.00        | No change                      |
| ddos      | 0.024       | 0.019       | ✅ Lower FNR → better detection |
| dos       | 0.042       | 0.034       | ✅ Improved                     |
| injection | 0.012       | 0.014       | Slightly worse                 |
| mitm      | 0.019       | 0.029       | Slightly worse                 |
| normal    | 0.033       | 0.013       | ✅ Significant improvement      |
| password  | 0.0055      | 0.0058      | Similar                        |
| scanning  | 0.0028      | 0.0030      | Similar                        |
| xss       | 0.024       | 0.012       | ✅ Improved                     |
