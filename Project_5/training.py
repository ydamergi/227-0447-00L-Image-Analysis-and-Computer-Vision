from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer,
    criterion,
    device,
    print_every_iters=100,
    save_path="./ckpt/model.pt"
):
    training_loss_per_epoch = []
    val_loss_per_epoch = []

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/ {num_epochs}")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % print_every_iters == 0:
                print(
                    f'[Epoch: {epoch + 1} / {num_epochs},'
                    f' Iter: {i + 1:5d} / {len(train_loader)}]'
                    f' Training loss: {running_loss / (i + 1):.3f}'
                )

        mean_loss = running_loss / len(train_loader)
        training_loss_per_epoch.append(mean_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader, desc="Validation")):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

        mean_loss = running_loss / len(val_loader)
        val_loss_per_epoch.append(mean_loss)

        if mean_loss < best_val_loss:
            best_val_loss = mean_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)

        print(
            f'[Epoch: {epoch + 1} / {num_epochs}]'
            f' Validation loss: {mean_loss:.3f}'
        )

    print(f'Finished Training. Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.3f}')

    # Plot the training curves
    plt.figure()
    plt.plot(np.array(training_loss_per_epoch))
    plt.plot(np.array(val_loss_per_epoch))
    plt.legend(['Training loss', 'Val loss'])
    plt.xlabel('Epoch')
    plt.show()
    plt.close()

    return {
        'training_loss': training_loss_per_epoch,
        'validation_loss': val_loss_per_epoch,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
