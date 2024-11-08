import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

def eval():
    true_labels_df = pd.read_csv("./test.csv")
    predicted_labels_df = pd.read_csv("./predicted.csv")

    label_encoder = LabelEncoder()
    true_labels = list(label_encoder.fit_transform(true_labels_df['Actual_Labels']))
    predicted_labels = list(label_encoder.transform(predicted_labels_df['Predicted_Labels']))
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    cm=confusion_matrix(true_labels,predicted_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
	eval()