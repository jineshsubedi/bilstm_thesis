"""
Copy and paste this code into a new cell in your Jupyter Notebook.
Hybrid V2:
1. 'dos': Aggressively Forces Recall >= 99.9% (Trying to get FNR near 0).
2. All others: Balanced (Youden's J).
"""

from sklearn.metrics import roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure model and data are loaded
# model, X_test_seq, y_test_seq, type_encoder

# 1. Get Predictions
print("Predicting on test set...")
y_pred_prob = model.predict(X_test_seq, verbose=0)
y_true = np.argmax(y_test_seq, axis=1)

# 2. Compute Hybrid Thresholds
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

# --- CONFIGURATION ---
DOS_TARGET_RECALL = 0.999  # 99.9% Recall target to minimize FNR
# ---------------------

print("\nComputing Hybrid V2 Thresholds (Aggressive DoS Recall)...")
print("-" * 105)
print(f"{'Class':<15} | {'Strategy':<15} | {'Threshold':<10} | {'TPR (Recall)':<15} | {'FPR':<15}")
print("-" * 105)

for i in range(n_classes):
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_binary)
    current_class = class_names[i]
    
    # --- STRATEGY ---
    if current_class == 'dos':
        strategy = f"Recall>={DOS_TARGET_RECALL}"
        # Find indices where Recall >= 0.999
        valid_indices = np.where(tpr >= DOS_TARGET_RECALL)[0]
        
        if len(valid_indices) > 0:
            # Pick highest threshold that meets criteria (minimize FPR damage)
            ix = valid_indices[0]
        else:
            # If 0.999 is never reached, take the max possible recall
            print(f"Warning: {current_class} could not reach {DOS_TARGET_RECALL} recall. Taking max possible.")
            ix = np.argmax(tpr)
            
        # If threshold is too high and we want to be safe, sometimes we pick slightly lower
        # But valid_indices[0] is usually the boundary.
        
    else:
        strategy = "Balanced"
        j_scores = tpr - fpr
        ix = np.argmax(j_scores)
        
    best_thresh = thresholds[ix]
    optimal_thresholds[i] = best_thresh
    
    # Safety clamp: if threshold is 0, we might predict everything. 
    # Let's keep a tiny epsilon if it's literally 0, unless intended.
    if optimal_thresholds[i] <= 0:
         optimal_thresholds[i] = 1e-9

    print(f"{current_class:<15} | {strategy:<15} | {best_thresh:.4f}     | {tpr[ix]:.4f}          | {fpr[ix]:.4f}")

# 3. Apply Thresholds
print("\n" + "-" * 60)
print("Applying Hybrid V2 thresholds...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    weighted_scores = [prob / optimal_thresholds[i] for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate
print("\nClassification Report (Hybrid V2):")
print("-" * 60)
print(classification_report(y_true, y_pred_optimal, target_names=class_names, digits=4))

# 5. Compute FPR/FNR
print("\nPer-Class FPR and FNR:")
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
plt.title(f'Confusion Matrix (Hybrid V2: DoS Recall >= {DOS_TARGET_RECALL})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
