import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm 
from utils.getData import create_dataloaders
from model.CNN import CNN

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(exist_ok=True)

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, val_loader, input_shape = create_dataloaders(
        data_path=PROCESSED_DATA_PATH / "detection_data.npz",
        batch_size=BATCH_SIZE
    )
    
    model = CNN(input_shape).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0, 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            predicted = torch.sigmoid(outputs) > 0.5
            train_correct += (predicted == labels.byte()).sum().item()
            
            train_loop.set_postfix(loss=batch_loss)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        model.eval()
        val_loss, val_correct = 0, 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                batch_loss = loss_fn(outputs, labels).item()
                val_loss += batch_loss
                predicted = torch.sigmoid(outputs) > 0.5
                val_correct += (predicted == labels.byte()).sum().item()
                
                val_loop.set_postfix(loss=batch_loss)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)
        
        print(
            f"Epoch Summary: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
        )

    print("\nTraining complete. Saving model state dictionary...")
    model_save_path = MODELS_PATH / "drone_detection_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")