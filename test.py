import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_and_plot(model, test_loader, history, device, results_path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(history['train_acc'], label='Train')
    axs[0].plot(history['val_acc'], label='Validation')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='upper left')
    axs[1].plot(history['train_loss'], label='Train')
    axs[1].plot(history['val_loss'], label='Validation')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    history_plot_path = results_path / "training_history_plot.png"
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    plt.show()

    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).long()
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTotal Accuracy on Test Set: {test_accuracy:.4f}")

    class_names = ['Non-Drone', 'Drone']
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_plot_path = results_path / "confusion_matrix.png"
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    plt.show()