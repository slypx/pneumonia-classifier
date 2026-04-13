# utils/train.py
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import LOG_DIR, TB_LOG_DIR, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_PATH
from .evaluate import evaluate_model

def train_model(model, train_loader, val_loader, device):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(TB_LOG_DIR)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        val_loss, val_acc, _ = evaluate_model(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(">> Saved new best model.")

    writer.close()

    # Save training history
    with open(LOG_DIR / "training_logs.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
