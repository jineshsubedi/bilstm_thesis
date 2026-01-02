"""
Copy and paste this code into a new cell in your Jupyter Notebook
(bayesian_optimized_model copy.ipynb) to compute thresholds optimizing for F2-SCORE.
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

# 2. Compute Optimal Thresholds per Class (MAXIMIZING F2-SCORE)
optimal_thresholds = {}
n_classes = y_test_seq.shape[1]
class_names = type_encoder.classes_

print("\nFinding optimal thresholds per class (Maximizing F2-Score for Lower FNR)...")
print("-" * 80)
print(f"{'Class':<15} | {'Threshold':<10} | {'Max F2':<10} | {'Recall (at Max F2)':<15}")
print("-" * 80)

for i in range(n_classes):
    # one-vs-rest binary labels
    y_true_binary = (y_true == i).astype(int)
    y_prob_binary = y_pred_prob[:, i]
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_prob_binary)
    
    # --- CHANGE: Calculate F2 Score instead of F1 ---
    # F2 weighs Recall higher (beta=2). beta=2 means recall is 2x as important as precision.
    beta = 2
    numerator = (1 + beta**2) * (precision * recall)
    denominator = (beta**2 * precision) + recall
    
    # Handle division by zero
    f2_scores = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator!=0)
    
    # Find index of max F2
    ix = np.argmax(f2_scores)
    
    # The thresholds array is one element shorter than precision/recall arrays
    if ix < len(thresholds):
        best_thresh = thresholds[ix]
        best_recall = recall[ix]
    else:
        best_thresh = thresholds[-1]
        best_recall = recall[-1]
        
    optimal_thresholds[i] = best_thresh
    max_f2 = f2_scores[ix] if ix < len(f2_scores) else 0.0
    
    # Force threshold to be at least small epsilon to avoid divide by zero later
    if optimal_thresholds[i] < 1e-9:
        optimal_thresholds[i] = 1e-9

    print(f"{class_names[i]:<15} | {best_thresh:.4f}     | {max_f2:.4f}     | {best_recall:.4f}")

# 3. Apply Thresholds for Prediction
print("\n" + "-" * 60)
print("Applying F2-optimal thresholds to predictions...")
y_pred_optimal = []

for sample_probs in y_pred_prob:
    # Strategy: Normalize probability by threshold => prob / threshold
    # The class that relatively exceeds its threshold the most wins.
    weighted_scores = [prob / optimal_thresholds[i] for i, prob in enumerate(sample_probs)]
    y_pred_optimal.append(np.argmax(weighted_scores))

y_pred_optimal = np.array(y_pred_optimal)

# 4. Evaluate and Plot Confusion Matrix
print("\nClassification Report (F2 Optimized Thresholds):")
print("-" * 60)
print(classification_report(y_true, y_pred_optimal, target_names=class_names, digits=4))

print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_optimal)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Optimized for F2-Score / Recall)')
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
