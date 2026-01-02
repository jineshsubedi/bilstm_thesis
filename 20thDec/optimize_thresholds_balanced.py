"""
Copy and paste this code into a new cell in your Jupyter Notebook.
This script finds thresholds using YOUDEN'S J STATISTIC to balance FPR and FNR.
Target: Minimize both (FPR <= 1% and FNR <= 1%) if possible.
"""

from sklearn.metrics import roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure you have your model and test data loaded
# model, X_test_seq, y_test_seq, type_encoder

# 1. Get Predictions
print("Predicting on test set...")
y_pred_prob = model.predict(X_test_seq, verbose=0)
y_true = np.argmax(y_test_seq, axis=1)

# 2. Compute Balanced Thresholds (Youden's Index)
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

print("\nFinding Balanced Thresholds (Youden's J: Maximize TPR - FPR)...")
print("-" * 105)
print(f"{'Class':<15} | {'Threshold':<10} | {'J Score':<10} | {'TPR (Recall)':<15} | {'FPR':<15} | {'FNR':<15}")
print("-" * 105)

for i in range(n_classes):
    # one-vs-rest binary labels
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    # Use ROC Curve to get TPR (Recall) and FPR for all thresholds
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_binary)
    
    # Calulcate Youden's J statistic = TPR - FPR
    # We want to MAXIMIZE this difference.
    j_scores = tpr - fpr
    
    ix = np.argmax(j_scores)
    best_thresh = thresholds[ix]
    
    # Metrics at this threshold
    best_tpr = tpr[ix]
    best_fpr = fpr[ix]
    best_fnr = 1 - best_tpr
    
    # --- Custom Logic for Hard Constraints ---
    # If standard Youden still gives bad FNR/FPR (> 1%), we can search for the "Cross-over" point
    # where FPR approx equals FNR, or check for nearest point to (0,1)
    
    # Let's stick to standard Youden first, as it's the mathematical definition of "Best Balance".
    
    optimal_thresholds[i] = best_thresh
    
    # Safety clamp
    if optimal_thresholds[i] < 1e-9:
        optimal_thresholds[i] = 1e-9

    print(f"{class_names[i]:<15} | {best_thresh:.4f}     | {j_scores[ix]:.4f}     | {best_tpr:.4f}          | {best_fpr:.4f}          | {best_fnr:.4f}")

# 3. Apply Thresholds for Prediction
print("\n" + "-" * 60)
print("Applying Balanced thresholds to predictions...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    # Strategy: Normalize probability by threshold
    weighted_scores = [prob / optimal_thresholds[i] for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate
print("\nClassification Report (Balanced Optimization):")
print("-" * 60)
print(classification_report(y_true, y_pred_optimal, target_names=class_names, digits=4))

# 5. Compute FPR and FNR per class
print("\nPer-Class False Positive Rate (FPR) and False Negative Rate (FNR):")
print("-" * 80)
print(f"{'Class':<15} | {'FPR':<10} | {'FNR':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
print("-" * 80)

cm = confusion_matrix(y_true, y_pred_optimal)
for i, class_label in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    print(f"{class_label:<15} | {FPR:.4f}     | {FNR:.4f}     | {TP:<6} | {FP:<6} | {FN:<6} | {TN:<6}")
print("-" * 80)

print("\nGenerating Confusion Matrix...")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix (Balanced Youden Thresholds)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
