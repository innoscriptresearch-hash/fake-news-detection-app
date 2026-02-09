import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Load Results
RESULTS_PATH = "fake_news_detection/results/microsoft_deberta-v3-base_results.json"

with open(RESULTS_PATH, "r") as f:
    results = json.load(f)

history = results["training_history"]

#Training & Validation Accuracy vs Epochs
epochs = [h["epoch"] for h in history]
train_acc = [h["train_accuracy"] for h in history]
val_acc = [h["val_accuracy"] for h in history]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, marker='o', label="Training Accuracy")
plt.plot(epochs, val_acc, marker='s', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Training Loss vs Epochs
train_loss = [h["train_loss"] for h in history]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', color='red')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.tight_layout()
plt.show()

#Overfitting Analysis with Best Epoch
best_epoch_idx = np.argmax(val_acc)
best_epoch = epochs[best_epoch_idx]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.axvline(x=best_epoch, linestyle="--", color="black",
            label=f"Best Epoch = {best_epoch}")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis and Best Epoch Selection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Confusion Matrix of the Proposed Model
conf_matrix = np.array(results["confusion_matrix"])
labels = ["Real", "Fake"]

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of the Proposed Model")
plt.tight_layout()
plt.show()

#Precision, Recall & F1-Score Comparison

metrics = {
    "Precision": results["precision"],
    "Recall": results["recall"],
    "F1-Score": results["f1_score"]
}

plt.figure(figsize=(7,5))
plt.bar(metrics.keys(), metrics.values(), color=["#2563eb", "#10b981", "#f59e0b"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score Comparison")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
