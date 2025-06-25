import contextlib
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from data.delay_simulation import simulate_random_delays, apply_fixed_delay_to_batch
from models.randomforest import RandomForestModel
import joblib

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip):
    """
    Train one epoch of the model.

    :param model: PyTorch model to be trained
    :param dataloader: Dataloader for training data
    :param optimizer: PyTorch optimizer (e.g. Adam)
    :param loss_fn: Binary classification loss function (e.g., BCELoss)
    :param device: torch.device ('cpu' or 'cuda')
    :param grad_clip: Gradient clipping value
    :return: Dictionary of averaged metrics: loss, precision, recall, f1
    """

    model.train() # Set model to training mode
    epoch_loss = 0
    all_predictions, all_labels, all_probs = [], [], []

    for x_batch, y_batch in dataloader:
        # Mode data to GPU/CPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad() # Reset gradients
        outputs = model(x_batch).squeeze() # Forward pass with shape [batch_size]

        loss = loss_fn(outputs, y_batch.squeeze()) # Compute binary cross-entropy
        loss.backward() # Backward pass

        # Gradient clipping to prevent exploding gradients
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step() # Update model weights

        epoch_loss += loss.item() # Accumulate loss

        # Threshold outputs to get binary predictions (0 or 1)
        predictions = (outputs.detach().cpu().numpy() > 0.5).astype(int)
        labels = y_batch.detach().cpu().numpy().astype(int)

        all_predictions.extend(predictions)
        all_labels.extend(labels)

    # Compute metrics for the full epoch
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = epoch_loss / len(dataloader) # Average loss

    return metrics

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute precision, recall, and F1 score.

    :param y_true: Ground-truth labels (list or array)
    :param y_pred: Predicted labels (list or array)
    :param y_probs: Predicted probabilities (list or array), optional
    :return: Dictionary of metrics: precision, recall, f1
    """

    # Handle cases where y_probs is None or has only one class
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        try:
            auc = roc_auc_score(y_true, y_probs) if y_probs is not None else None
        except ValueError:
            auc = None

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0), # How correct the positive predictions are
        "recall": recall_score(y_true, y_pred, zero_division=0), # How many of the actual positives were predicted correctly
        "f1": f1_score(y_true, y_pred, zero_division=0), # Harmonic mean of precision and recall
        "accuracy": accuracy_score(y_true, y_pred), # Overall accuracy of the model (True Positives + True Negatives) / Total
        "aur_roc": auc,
        # Area Under the Receiver Operating Characteristic Curve. Handle cases of 1D and 2D arrays
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist() # True Positives, False Positives, True Negatives, False Negatives
    }

def evaluate_model(
        model,
        dataloader,
        loss_fn,
        device,
        split_name="train",
        delay_steps=0,
        simulate_random=False,
        strategy="shift"
):
    """
    Evaluate the model on the validation or test set.

    :param strategy: Strategy to apply the delay. Option is only "shift" at the moment.
    :param simulate_random: If True, simulate random delays for test data.
    :param delay_steps: Number of delay steps to apply to the test data.
    :param split_name: Name of the split being evaluated (e.g., "train", "val", "test"), used for delay simulation.
    :param model: Trained PyTorch model
    :param dataloader: Dataloader for validation/test data
    :param loss_fn: Loss function for evaluation
    :param device: torch.device
    :return: Dictionary of averaged metrics: loss, precision, recall, f1
    """
    is_torch = isinstance(model, torch.nn.Module) or isinstance(model, torch.nn.modules.module.Module) or callable(model)

    if is_torch:
        model.eval() # Set model to evaluation mode
    total_loss = 0
    all_predictions, all_labels, all_probs = [], [], []

    with torch.no_grad() if is_torch else contextlib.nullcontext() : # Disable gradient calculation for evaluation
        for x_batch, y_batch in dataloader:
            if is_torch : # For PyTorch models
                # Move data to GPU/CPU
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if split_name == "test":
                    if simulate_random:
                        # Simulate random delays for test data
                        x_batch = simulate_random_delays(x_batch, 1, delay_steps)
                    else:
                        # Apply fixed delay to test data
                        x_batch = apply_fixed_delay_to_batch(x_batch, delay_steps, strategy=strategy)

                # Forward pass
                outputs = model(x_batch).squeeze() # Forward pass
                loss = loss_fn(outputs, y_batch.squeeze())

                total_loss += loss.item()

                probs = outputs.detach().cpu().numpy()
                predictions = (probs > 0.5).astype(int)
                labels = y_batch.detach().cpu().numpy().astype(int)

                all_probs.extend(probs)
                all_predictions.extend(predictions)
                all_labels.extend(labels)

            else: # For non-PyTorch models (e.g., RandomForest)
                x_np = x_batch.cpu().numpy()
                y_np = y_batch.numpy()

                probs = model.predict_proba(x_np)
                predictions = model.predict(x_np)

                all_probs.extend(probs)
                all_predictions.extend(predictions)
                all_labels.extend(y_np)

        metrics = compute_metrics(all_labels, all_predictions, y_probs=all_probs)
        if isinstance(model, torch.nn.Module):
            metrics['loss'] = total_loss / len(dataloader)
        else:
            metrics['loss'] = None
        return metrics

def train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        device,
        num_epochs,
        grad_clip,
        save_path="best_model.pth"
):
    """
    Full training loop with validation and checkpoint saving

    :param model: PyTorch model to be trained
    :param train_dataloader: Dataloader for training data
    :param val_dataloader: Dataloader for validation data
    :param optimizer: PyTorch optimizer
    :param loss_fn: Loss function (e.g. BCELoss)
    :param device: torch.device
    :param num_epochs: Number of epochs to train
    :param save_path: File path to save best model
    :param grad_clip: Gradient clipping value
    :return: History of training and validation metrics
    """

    best_f1 = 0
    patience = 5 # Early stopping patience
    patience_counter = 0 # Counter for early stopping
    history = {"train": [], "val": []}

    # Special handling for RandomForest
    if isinstance(model, RandomForestModel):
        x_train, y_train = train_dataloader.dataset.feature_array, train_dataloader.dataset.label_array
        x_val, y_val = val_dataloader.dataset.feature_array, val_dataloader.dataset.label_array

        seq_len = model.seq_len
        input_dim = model.input_dim

        # Reshape training data
        num_train_seq = x_train.shape[0] // seq_len
        x_train = x_train[:num_train_seq * seq_len].reshape(num_train_seq, seq_len, input_dim)
        y_train = y_train[:num_train_seq * seq_len:seq_len]

        # Reshape validation data
        num_val_seq = x_val.shape[0] // seq_len
        x_val = x_val[:num_val_seq * seq_len].reshape(num_val_seq, seq_len, input_dim)
        y_val = y_val[:num_val_seq * seq_len:seq_len]

        model.fit(x_train, y_train)

        val_predictions = model.predict(x_val)
        val_probs = model.predict_proba(x_val)

        val_metrics = compute_metrics(y_val, val_predictions, y_probs=val_probs)
        history["val"].append(val_metrics)

        print("RandomForest trained and evaluated.")
        # Save the model
        joblib.dump(model.model, save_path + ".joblib")
        print(f"RandomForest model saved to {save_path}.")
        return history

    # Training loop for PyTorch models
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device, grad_clip)
        val_metrics = evaluate_model(model, val_dataloader, loss_fn, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Save the model if the validation F1 score is the best so far
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            if save_path:
                torch.save(model.state_dict(), save_path + ".pt")
            print(f"  New best model saved to {save_path} with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    return history