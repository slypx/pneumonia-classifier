# utils/evaluate.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    print("Accuracy:", acc)
    print("F1-score:", f1)
    print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification report:\n", classification_report(all_labels, all_preds))

    return avg_loss, acc, f1
