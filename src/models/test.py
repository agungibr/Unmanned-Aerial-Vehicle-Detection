import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from src.utils.getData import create_dataloaders
from src.models.CNN import CNN

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['detection', 'distance'],
                        help="Type of model to evaluate: 'detection' or 'distance'.")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    with open(PROJECT_ROOT / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    params = config[args.model_type]
    default_params = config['default']

    PROCESSED_DATA_PATH = PROJECT_ROOT / default_params['data_path']
    MODELS_PATH = PROJECT_ROOT / default_params['model_save_path']
    RESULTS_PATH = PROJECT_ROOT / default_params['results_save_path']
    
    print(f"Evaluating {args.model_type.upper()} Model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader, input_shape, num_classes, class_map = create_dataloaders(
        processed_data_path=PROCESSED_DATA_PATH,
        data_filename=params['data_filename'],
        batch_size=default_params['batch_size']
    )

    model = CNN(input_shape, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODELS_PATH / params['model_filename'], map_location=device))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if args.model_type == 'detection':
                predicted = (torch.sigmoid(outputs) > 0.5).long()
            else: 
                predicted = torch.argmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTotal Accuracy on Test Set: {test_accuracy:.4f}")
    
    if args.model_type == 'detection':
        class_names = params['class_names']
    else: 
        class_names = sorted(class_map, key=class_map.get)
        
    plot_save_path = RESULTS_PATH / f"{args.model_type}_confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_preds, class_names, plot_save_path)