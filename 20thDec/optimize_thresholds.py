"""
Copy and paste this code into a new cell in your Jupyter Notebook
(bayesian_optimized_model copy.ipynb) to compute class-specific thresholds,
evaluate the model performance, and calculate FPR/FNR.
"""

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure you have your model and test data loaded/processed in the notebook variables:
# model, X_test_seq, y_test_seq, type_encoder

# 1. Get Predictions
print("Predicting on test set...")
y_pred_prob = model.predict(X_test_seq, verbose=0)
y_true = np.argmax(y_test_seq, axis=1)

# 2. Compute Optimal Thresholds per Class
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

print("\nFinding optimal thresholds per class (maximizing F1-score)...")
print("-" * 60)
for i in range(n_classes):
    # one-vs-rest binary labels
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_binary)
    
    # Calculate F1 for each threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    denom = (precision + recall)
    f1_scores = np.divide(2 * (precision * recall), denom, out=np.zeros_like(denom), where=denom!=0)
    
    # Find index of max F1
    ix = np.argmax(f1_scores)
    
    # The thresholds array is one element shorter than precision/recall arrays
    if ix < len(thresholds):
        best_thresh = thresholds[ix]
    else:
        best_thresh = thresholds[-1]
        
    optimal_thresholds[i] = best_thresh
    max_f1 = f1_scores[ix] if ix < len(f1_scores) else 0.0
    print(f"Class {i:<2} {class_names[i]:<12} | Threshold: {best_thresh:.4f} | Max F1: {max_f1:.4f}")

# 3. Apply Thresholds for Prediction
print("\n" + "-" * 60)
print("Applying optimal thresholds to predictions...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    # Strategy: Normalize probability by threshold => prob / threshold
    # The class that relatively exceeds its threshold the most wins.
    weighted_scores = [prob / (optimal_thresholds[i] + 1e-9) for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate and Plot Confusion Matrix
print("\nClassification Report (Optimized Thresholds):")
print("-" * 60)
print(classification_report(y_true, y_pred_optimal, target_names=class_names, digits=4))

print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_optimal)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix with Optimal Thresholds')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 5. Compute FPR and FNR per class
print("\nPer-Class False Positive Rate (FPR) and False Negative Rate (FNR):")
print("-" * 80)
print(f"{'Class':<15} | {'FPR':<10} | {'FNR':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
print("-" * 80)

metrics_list = []

for i, class_label in enumerate(class_names):
    # True Positives
    TP = cm[i, i]
    # False Positives: sum of column i - TP
    FP = cm[:, i].sum() - TP
    # False Negatives: sum of row i - TP
    FN = cm[i, :].sum() - TP
    # True Negatives: total sum - (TP + FP + FN)
    TN = cm.sum() - (TP + FP + FN)
    
    # Calculate Rates
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    metrics_list.append({
        'Class': class_label,
        'FPR': FPR,
        'FNR': FNR,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    })
    
    print(f"{class_label:<15} | {FPR:.4f}     | {FNR:.4f}     | {TP:<6} | {FP:<6} | {FN:<6} | {TN:<6}")

print("-" * 80)
