



"""
plot_training_and_validation_history

This script loads a Keras training history JSON file and plots training/validation
accuracy and loss across epochs. The plot is saved as a high-resolution image.

Author: Your Name
Date: 2025-05-01
"""

import json
import os
import argparse
import matplotlib.pyplot as plt


def plot_history(history_path, output_path):
    """
    Load training history and generate a combined plot of accuracy and loss.

    Args:
        history_path (str): Path to the JSON history file.
        output_path (str): Path to save the output plot image.
    """
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, 'r') as f:
        history = json.load(f)

    train_acc = history.get('categorical_accuracy', [])
    val_acc = history.get('val_categorical_accuracy', [])
    train_loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    if not train_acc or not val_acc or not train_loss or not val_loss:
        raise ValueError("Missing keys in history file. Required: 'categorical_accuracy', 'val_categorical_accuracy', 'loss', 'val_loss'.")

    epochs = range(1, len(train_acc) + 1)
    min_val_loss_epoch = val_loss.index(min(val_loss)) + 1
    min_val_loss = min(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'r--', label='Train Loss')
    plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
    plt.plot(epochs, train_acc, 'b--', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy')

    # Highlight the best model (lowest validation loss)
    plt.scatter(min_val_loss_epoch, min_val_loss, color='purple',
                s=100, label=f'Best Model (Epoch {min_val_loss_epoch})', edgecolors='black')

    plt.title('Training and Validation Accuracy and Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Metrics (Accuracy / Loss)', fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=11)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Plot saved to {output_path}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation accuracy/loss from Keras history JSON.")
    parser.add_argument('--history_path', required=True, help='Path to the training history JSON file')
    parser.add_argument('--output_path', required=True, help='Path to save the output plot (e.g., plot.jpg)')

    args = parser.parse_args()
    plot_history(args.history_path, args.output_path)


if __name__ == "__main__":
    main()

