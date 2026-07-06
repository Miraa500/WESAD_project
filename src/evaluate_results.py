import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import os

os.makedirs("results", exist_ok=True)
  

X = np.load("data/X.npy")
y = np.load("data/y.npy")
subjects = np.load("data/subjects.npy")
    
le = LabelEncoder()
y_encoded = le.fit_transform(y)

TEST_SUBJECTS = ["S15", "S16", "S17"]   # لازم يبقوا نفس الأسماء المستخدمة في model.py

test_mask = np.isin(subjects, TEST_SUBJECTS)
X_test, y_test = X[test_mask], y_encoded[test_mask]

print("Test set:", X_test.shape, "| Subjects:", np.unique(subjects[test_mask]))


model = tf.keras.models.load_model("cnn_wesad_model.h5")



y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


test_accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy:", test_accuracy)



label_map = {0: "Baseline", 1: "Stress", 2: "Amusement"}  # حسب ترتيب LabelEncoder لـ [1,2,3]
target_names = [label_map[i] for i in sorted(np.unique(y_test))]

report_dict = classification_report(
    y_test, y_pred,
    target_names=target_names,
    output_dict=True
)

report_text = classification_report(
    y_test, y_pred,
    target_names=target_names
)

print("\nClassification Report:\n")
print(report_text)

with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

with open("results/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=4)


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names
)
plt.title("Confusion Matrix - Test Set (Subject-Independent)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()



summary = {
    "test_accuracy": float(test_accuracy),
    "test_subjects": TEST_SUBJECTS,
    "num_test_samples": int(X_test.shape[0]),
    "classes": target_names
}

with open("results/final_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
    
print(" - classification_report.txt / .json")
print(" - confusion_matrix.png")
print(" - final_summary.json")
