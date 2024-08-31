import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrices(cm_enkf, cm_adam, classes, title_enkf="Confusion Matrix EnKF", title_adam="Confusion Matrix Adam"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.get_cmap("Blues")

    # Plot Confusion Matrix for EnKF
    sns.heatmap(cm_enkf, annot=True, fmt='d', cmap=cmap, cbar=False, ax=axes[0], xticklabels=classes, yticklabels=classes)
    axes[0].set_title(title_enkf)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Plot Confusion Matrix for Adam
    sns.heatmap(cm_adam, annot=True, fmt='d', cmap=cmap, cbar=False, ax=axes[1], xticklabels=classes, yticklabels=classes)
    axes[1].set_title(title_adam)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()

def print_classification_reports():
    report_enkf = """
    Classification Report for EnKF:
                   precision    recall  f1-score   support

         Class 0       0.68      0.90      0.78        21
         Class 1       0.93      0.62      0.74        21
         Class 2       0.80      0.57      0.67        21
         Class 3       0.70      0.78      0.74        18
         Class 4       0.62      0.75      0.68        20

        accuracy                           0.72       101
       macro avg       0.75      0.72      0.72       101
    weighted avg       0.75      0.72      0.72       101
    """

    report_adam = """
    Classification Report for Adam:
                   precision    recall  f1-score   support

         Class 0       0.80      0.95      0.87        21
         Class 1       0.86      0.86      0.86        21
         Class 2       0.94      0.81      0.87        21
         Class 3       0.79      0.61      0.69        18
         Class 4       0.74      0.85      0.79        20

        accuracy                           0.82       101
       macro avg       0.83      0.82      0.82       101
    weighted avg       0.83      0.82      0.82       101
    """
    
    print(report_enkf)
    print(report_adam)

# Confusion matrices
cm_enkf = np.array([[19,  0,  1,  1,  0],
                    [ 1, 13,  2,  3,  2],
                    [ 3,  0, 12,  0,  6],
                    [ 3,  0,  0, 14,  1],
                    [ 2,  1,  0,  2, 15]])

cm_adam = np.array([[20,  1,  0,  0,  0],
                    [ 1, 18,  1,  1,  0],
                    [ 1,  0, 17,  0,  3],
                    [ 3,  1,  0, 11,  3],
                    [ 0,  1,  0,  2, 17]])

classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

# Plot the confusion matrices
plot_confusion_matrices(cm_enkf, cm_adam, classes)

# Print the classification reports
print_classification_reports()
