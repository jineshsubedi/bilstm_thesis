"""
Copy and paste this code into a new cell in your Jupyter Notebook.
This script uses a HYBRID approach:
1. Uses Youden's J (Balanced) for most classes.
2. Specifically OVERRIDES the 'dos' class threshold to force lower FNR.
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

# 2. Compute Hybrid Thresholds
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

print("\nComputing Hybrid Thresholds...")
print("-" * 105)
print(f"{'Class':<15} | {'Strategy':<10} | {'Threshold':<10} | {'TPR (Recall)':<15} | {'FPR':<15}")
print("-" * 105)

for i in range(n_classes):
    # one-vs-rest binary labels
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_binary)
    
    current_class = class_names[i]
    
    # --- STRATEGY SELECTION ---
    if current_class == 'dos':
        # SPECIAL HANDLING FOR DOS: Force FNR <= 1% (Recall >= 99%)
        strategy = "Force Recall"
        # Find first threshold where TPR >= 0.99
        # thresholds are decreasing in roc_curve usually? verify.
        # sklearn roc_curve thresholds are decreasing.
        
        # We search for the *highest* threshold that still gives TPR >= 0.99
        valid_indices = np.where(tpr >= 0.99)[0]
        if len(valid_indices) > 0:
            # First index in valid_indices corresponds to the highest threshold (lowest FPR) 
            # that satisfies the recall constraint.
            ix = valid_indices[0]
        else:
            # Fallback to max recall if 0.99 is unreachable
            ix = np.argmax(tpr)
            
    else:
        # DEFAULT STRATEGY: Youden's J (Balanced)
        strategy = "Balanced"
        j_scores = tpr - fpr
        ix = np.argmax(j_scores)
        
    best_thresh = thresholds[ix]
    optimal_thresholds[i] = best_thresh
    
    # Safety clamp
    if optimal_thresholds[i] < 1e-9:
        optimal_thresholds[i] = 1e-9

    print(f"{current_class:<15} | {strategy:<10} | {best_thresh:.4f}     | {tpr[ix]:.4f}          | {fpr[ix]:.4f}")

# 3. Apply Hybrid Thresholds
print("\n" + "-" * 60)
print("Applying Hybrid thresholds to predictions...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    weighted_scores = [prob / optimal_thresholds[i] for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate
print("\nClassification Report (Hybrid Optimization):")
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
plt.title(f'Confusion Matrix (Hybrid: Balanced + Strict DoS)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
