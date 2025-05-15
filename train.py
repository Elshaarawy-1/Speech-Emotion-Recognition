import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from utils import get_dataset, create_model
import parser

def train(train_loader, val_loader, model, optimizer, criterion, epochs=10):
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).long()
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"saved_model/{epochs}_trained_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Plot loss curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/loss_curve_{epochs}.png")
    plt.close()

def test(model, test_loader, criterion, epochs):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device).float(), y.to(device).long()
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item() * x.size(0)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # F1-Score and Confusion Matrix
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"F1 Score (weighted): {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"outputs/confusion_matrix_{epochs}.png")
    plt.close()

if __name__ == "__main__":
    args = parser.default_parser()

    train_loader, val_loader, test_loader = get_dataset(
        batch_size=64,
        random_state=42,
        num_workers=4,
    )

    model = create_model(args.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train(train_loader, val_loader, model, optimizer, criterion, epochs=args.epochs)

    print("\nEvaluating on test set...")
    test(model, test_loader, criterion, args.epochs)