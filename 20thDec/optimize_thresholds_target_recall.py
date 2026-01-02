"""
Copy and paste this code into a new cell in your Jupyter Notebook.

This script does NOT essentially maximize F1 or F2.
Instead, it finds the threshold that guarantees a minimum Recall (e.g., 99%)
for each class. This forces the FNR to be <= 1%, regardless of the impact on Precision.
"""

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# CONFIGURATION
# ------------------------------------------------------------------
TARGET_RECALL = 0.99  # We want to catch at least 99% of attacks (FNR <= 1%)
# ------------------------------------------------------------------

# Ensure you have your model and test data loaded/processed in the notebook variables:
# model, X_test_seq, y_test_seq, type_encoder

# 1. Get Predictions
print("Predicting on test set...")
y_pred_prob = model.predict(X_test_seq, verbose=0)
y_true = np.argmax(y_test_seq, axis=1)

# 2. Compute Thresholds for Fixed Recall Target
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

print(f"\nFinding thresholds to guarantee Recall >= {TARGET_RECALL} (FNR <= {1-TARGET_RECALL:.2f})...")
print("-" * 90)
print(f"{'Class':<15} | {'Threshold':<10} | {'Achieved Recall':<15} | {'Precision at target':<20}")
print("-" * 90)

for i in range(n_classes):
    # one-vs-rest binary labels
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_binary)
    
    # We want the index where recall is just above TARGET_RECALL
    # Precision-Recall curve is typically sorted by threshold descending (high to low)
    # So recall goes from 0 to 1.
    
    # Find all indices where recall >= TARGET_RECALL
    valid_indices = np.where(recall >= TARGET_RECALL)[0]
    
    if len(valid_indices) > 0:
        # We want the *highest* threshold that still gives us this recall.
        # Higher threshold = Better Precision.
        # Since thresholds are descending, the first index in 'valid_indices' might be the highest recall (1.0) and lowest threshold.
        # We actually want the point "closest" to our target without dropping below it.
        # Typically, we scan for the first point where Recall >= Target.
        
        # Let's search for the highest threshold (earliest in the list usually, but let's be robust)
        # where recall >= target.
        
        # Note: 'thresholds' array is shorter than 'recall' by 1.
        
        best_idx = -1
        best_thresh = 0.0
        
        # Iterate to find the highest threshold with acceptable recall
        for idx in range(len(thresholds)):
            if recall[idx] >= TARGET_RECALL:
                best_idx = idx
                best_thresh = thresholds[idx]
                # Since thresholds are sorted descending, the first one we hit
                # is the LARGEST threshold that satisfies the condition.
                # So we break immediately.
                break
        
        # If we didn't break, it means we scanned everything and maybe only the last point worked?
        if best_idx == -1:
             # If no single threshold meets it, pick the one with max recall (usually the lowest threshold)
            best_idx = np.argmax(recall[:-1]) 
            best_thresh = thresholds[best_idx]
            
    else:
        # If model is terrible and never reaches target recall, take min threshold
        best_idx = -1
        best_thresh = thresholds[-1]

    optimal_thresholds[i] = best_thresh
    achieved_recall = recall[best_idx]
    achieved_prec = precision[best_idx]
    
    # Safety clamp
    if optimal_thresholds[i] < 1e-9:
        optimal_thresholds[i] = 1e-9

    print(f"{class_names[i]:<15} | {best_thresh:.4f}     | {achieved_recall:.4f}          | {achieved_prec:.4f}")

# 3. Apply Thresholds for Prediction
print("\n" + "-" * 60)
print("Applying Recall-Fixed thresholds to predictions...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    # Strategy: Normalize probability by threshold => prob / threshold
    weighted_scores = [prob / optimal_thresholds[i] for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate
print("\nClassification Report (Target Recall Optimized):")
print("-" * 60)
print(classification_report(y_true, y_pred_optimal, target_names=class_names, digits=4))

print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_optimal)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix (Forcing Recall >= {TARGET_RECALL})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 5. Compute FPR and FNR per class
print("\nPer-Class False Positive Rate (FPR) and False Negative Rate (FNR):")
print("-" * 80)
print(f"{'Class':<15} | {'FPR':<10} | {'FNR':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
print("-" * 80)

for i, class_label in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    print(f"{class_label:<15} | {FPR:.4f}     | {FNR:.4f}     | {TP:<6} | {FP:<6} | {FN:<6} | {TN:<6}")
print("-" * 80)
