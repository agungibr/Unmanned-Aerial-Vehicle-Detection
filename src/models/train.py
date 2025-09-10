import torch
import torch.nn as nn
import yaml
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from ..utils.getData import create_dataloaders
from .CNN import CNN
from .MLP import MLP

def plot_training_history(history, save_path):
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
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model based on config.")-
    parser.add_argument('experiment', type=str, required=True, 
                        choices=['detection', 'distance_mlp'],
                        help="Name of the experiment to run from config.yaml.")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    with open(PROJECT_ROOT / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    params = config[args.experiment]
    default_params = config['default']
    
    PROCESSED_DATA_PATH = PROJECT_ROOT / default_params['data_path']
    MODELS_PATH = PROJECT_ROOT / default_params['model_save_path']
    RESULTS_PATH = PROJECT_ROOT / default_params['results_save_path']
    MODELS_PATH.mkdir(exist_ok=True)
    RESULTS_PATH.mkdir(exist_ok=True)

    LEARNING_RATE = params['learning_rate']
    BATCH_SIZE = default_params['batch_size']
    EPOCHS = params['epochs']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {args.experiment.upper()} Model on {device}")

    train_loader, val_loader, _, input_shape, num_classes, _ = create_dataloaders(
        processed_data_path=PROCESSED_DATA_PATH,
        data_filename=params['data_filename'],
        batch_size=BATCH_SIZE
    )
    
    if params['model_architecture'] == 'cnn':
        model = CNN(input_shape, num_classes=num_classes).to(device)
    elif params['model_architecture'] == 'mlp':
        input_size = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        model = MLP(input_size=input_size, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model architecture: {params['model_architecture']}")

    loss_fn = nn.BCEWithLogitsLoss() if args.experiment == 'detection' else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if args.experiment == 'detection':
                loss = loss_fn(outputs, labels.float().unsqueeze(1))
                predicted = torch.sigmoid(outputs) > 0.5
                train_correct += (predicted == labels.byte().unsqueeze(1)).sum().item()
            else: 
                loss = loss_fn(outputs, labels)
                predicted = torch.argmax(outputs, dim=1)
                train_correct += (predicted == labels).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if args.experiment == 'detection':
                    val_loss += loss_fn(outputs, labels.float().unsqueeze(1)).item()
                    predicted = torch.sigmoid(outputs) > 0.5
                    val_correct += (predicted == labels.byte().unsqueeze(1)).sum().item()
                else:
                    val_loss += loss_fn(outputs, labels).item()
                    predicted = torch.argmax(outputs, dim=1)
                    val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)
        
        print(f"Epoch Summary: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

    model_save_path = MODELS_PATH / params['model_filename']
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_save_path = RESULTS_PATH / f"{args.experiment}_history_plot.png"
    plot_training_history(history, plot_save_path)